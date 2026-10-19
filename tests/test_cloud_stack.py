"""tests/test_cloud_stack.py — safety properties of cloud/stack.yaml and requirements-cloud.txt (no AWS calls)."""

from pathlib import Path
from typing import Any

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent


class CfnLoader(yaml.SafeLoader):
    """SafeLoader that turns CloudFormation short tags (!Ref, !Sub, ...) into plain {tag: value} dicts."""


def _tag(loader: CfnLoader, suffix: str, node: yaml.Node) -> Any:
    if isinstance(node, yaml.ScalarNode):
        value: Any = loader.construct_scalar(node)
    elif isinstance(node, yaml.SequenceNode):
        value = loader.construct_sequence(node, deep=True)
    else:
        value = loader.construct_mapping(node, deep=True)
    return {suffix: value}


CfnLoader.add_multi_constructor("!", _tag)


@pytest.fixture(scope="module")
def stack() -> dict[str, Any]:
    return yaml.load((ROOT / "cloud" / "stack.yaml").read_text(), Loader=CfnLoader)


def _resources(stack: dict[str, Any], rtype: str) -> list[dict[str, Any]]:
    return [r for r in stack["Resources"].values() if r["Type"] == rtype]


def test_launch_template_is_spot_c7a_and_self_terminating(stack: dict[str, Any]) -> None:
    (lt,) = _resources(stack, "AWS::EC2::LaunchTemplate")
    data = lt["Properties"]["LaunchTemplateData"]
    assert data["InstanceType"] == "c7a.16xlarge"
    assert data["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert data["InstanceMarketOptions"]["MarketType"] == "spot"
    assert data["InstanceMarketOptions"]["SpotOptions"]["InstanceInterruptionBehavior"] == "terminate"
    assert data["InstanceMarketOptions"]["SpotOptions"]["SpotInstanceType"] == "one-time"
    assert data["MetadataOptions"]["HttpTokens"] == "required"  # IMDSv2
    assert data["BlockDeviceMappings"][0]["Ebs"]["Encrypted"] is True
    assert data["BlockDeviceMappings"][0]["Ebs"]["DeleteOnTermination"] is True
    assert "UserData" not in data  # the launcher passes bucket/job/deadline at run time


def test_security_group_has_no_inbound(stack: dict[str, Any]) -> None:
    (sg,) = _resources(stack, "AWS::EC2::SecurityGroup")
    assert sg["Properties"].get("SecurityGroupIngress", []) == []


def test_bucket_is_private_encrypted_and_https_only(stack: dict[str, Any]) -> None:
    (bucket,) = _resources(stack, "AWS::S3::Bucket")
    props = bucket["Properties"]
    assert all(props["PublicAccessBlockConfiguration"].values()) and len(props["PublicAccessBlockConfiguration"]) == 4
    assert props["BucketEncryption"]
    assert bucket["DeletionPolicy"] == "Retain" and bucket["UpdateReplacePolicy"] == "Retain"
    (policy,) = _resources(stack, "AWS::S3::BucketPolicy")
    (stmt,) = policy["Properties"]["PolicyDocument"]["Statement"]
    assert stmt["Effect"] == "Deny" and stmt["Condition"]["Bool"]["aws:SecureTransport"] == "false"


def _agent_statements(stack: dict[str, Any]) -> list[dict[str, Any]]:
    (policy,) = _resources(stack, "AWS::IAM::ManagedPolicy")
    return policy["Properties"]["PolicyDocument"]["Statement"]


def _as_list(x: Any) -> list[Any]:
    return x if isinstance(x, list) else [x]


def test_agent_policy_has_no_wildcard_actions_or_resources(stack: dict[str, Any]) -> None:
    for stmt in _agent_statements(stack):
        assert stmt["Effect"] == "Allow"
        for action in _as_list(stmt["Action"]):
            assert "*" not in action, stmt["Sid"]
        for resource in _as_list(stmt["Resource"]):
            assert resource != "*", stmt["Sid"]


def test_agent_policy_pins_instance_type_and_scopes_terminate_to_project_tag(stack: dict[str, Any]) -> None:
    by_sid = {s["Sid"]: s for s in _agent_statements(stack)}
    # only 64-core AMD Zen4 types (same CPU as c7a, so results stay comparable); launch.py picks the cheapest Spot pool among them
    assert by_sid["RunOnlyZen4Sixteenxlarge"]["Condition"]["StringEquals"]["ec2:InstanceType"] == ["c7a.16xlarge", "m7a.16xlarge", "r7a.16xlarge"]
    assert "ec2:ResourceTag/project" in by_sid["TerminateProjectInstances"]["Condition"]["StringEquals"]
    assert "ssm:resourceTag/project" in by_sid["SsmTargetProjectInstancesOnly"]["Condition"]["StringEquals"]
    assert by_sid["PassOnlyTheInstanceRole"]["Condition"]["StringEquals"]["iam:PassedToService"] == "ec2.amazonaws.com"


def test_agent_policy_grants_only_the_expected_actions(stack: dict[str, Any]) -> None:
    granted = {a for s in _agent_statements(stack) for a in _as_list(s["Action"])}
    assert granted == {
        "ec2:RunInstances", "ec2:CreateTags", "ec2:TerminateInstances", "iam:PassRole", "ssm:SendCommand", "s3:PutObject", "s3:DeleteObject",
    }  # fmt: skip


def test_instance_role_s3_access_is_limited_to_results_bucket(stack: dict[str, Any]) -> None:
    (role,) = _resources(stack, "AWS::IAM::Role")
    (inline,) = role["Properties"]["Policies"]
    for stmt in inline["PolicyDocument"]["Statement"]:
        assert all(a.startswith("s3:") for a in _as_list(stmt["Action"]))
        assert "*" not in "".join(_as_list(stmt["Action"]))


def test_requirements_cloud_pins_every_package() -> None:
    lines = [ln.split("#")[0].strip() for ln in (ROOT / "requirements-cloud.txt").read_text().splitlines()]
    reqs = [ln for ln in lines if ln]
    assert reqs, "requirements-cloud.txt is empty"
    assert all("==" in ln for ln in reqs), [ln for ln in reqs if "==" not in ln]
    assert any(ln.startswith("torch==") for ln in reqs) and any(ln.startswith("pooltool-billiards==") for ln in reqs)


def test_setup_script_installs_torch_cpu_first_and_provides_panda3d_dev_index() -> None:
    """pooltool 0.5.0 pins panda3d==1.11.0.dev3702 on Linux (not on PyPI); the first cloud smoke job failed without the index."""
    lines = (ROOT / "cloud" / "setup_cloud.sh").read_text().splitlines()
    install = [ln for ln in lines if ln.lstrip().startswith("uv pip install")]
    assert len(install) == 2
    assert "download.pytorch.org/whl/cpu" in install[0] and "torch==2.10.0" in install[0]  # CPU wheel before everything else
    assert "--extra-index-url https://archive.panda3d.org/" in install[1] and "-r requirements-cloud.txt" in install[1]
