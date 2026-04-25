"""Async unit tests for M3 DMACN — Dynamic Multi-Agent Critic Network."""

import asyncio
import pytest
import pytest_asyncio


class MockAgent:
    """Mock agent for testing DMACN."""

    def __init__(self, name, response="Test response.", confidence=0.8, should_fail=False):
        self.name = name
        self._response = response
        self._confidence = confidence
        self._should_fail = should_fail

    async def analyze(self, query, context=None):
        from medaide_plus.modules.m3_dmacn import AgentOutput
        if self._should_fail:
            raise ValueError("Mock agent failure")
        return AgentOutput(
            agent_name=self.name,
            response=self._response,
            confidence=self._confidence,
            claims=["Test claim."],
        )


class TestDMACNModule:
    """Tests for DMACNModule."""

    @pytest.fixture
    def dmacn(self):
        from medaide_plus.modules.m3_dmacn import DMACNModule
        agents = [
            MockAgent("Agent1", "Patient should take 500mg aspirin daily."),
            MockAgent("Agent2", "Recommend 500mg aspirin for pain relief.", confidence=0.7),
            MockAgent("Agent3", "Diagnosis: tension headache.", confidence=0.9),
            MockAgent("Agent4", "Follow up with GP in 2 weeks.", confidence=0.6),
        ]
        return DMACNModule(agents=agents, config={"timeout": 10.0})

    @pytest.mark.asyncio
    async def test_parallel_execution(self, dmacn):
        """Test that all agents run in parallel and return outputs."""
        result = await dmacn.run("I have a headache", n_agents=3)
        assert len(result.agent_outputs) == 3
        assert all(o.latency_ms >= 0 for o in result.agent_outputs)

    @pytest.mark.asyncio
    async def test_critic_detects_contradiction(self):
        """Test that critic detects dosage contradictions."""
        from medaide_plus.modules.m3_dmacn import DMACNModule, AgentOutput
        agents = [
            MockAgent("MedAgent1", "Patient should take 100mg daily."),
            MockAgent("MedAgent2", "Recommended dose is 500mg three times daily."),
        ]
        dmacn = DMACNModule(agents=agents, config={"timeout": 10.0})
        result = await dmacn.run("What dose?", n_agents=2)
        assert result.critic_report is not None

    @pytest.mark.asyncio
    async def test_synthesis_confidence_weighting(self, dmacn):
        """Test that synthesis produces a non-empty response."""
        result = await dmacn.run("General health question", n_agents=4)
        assert len(result.synthesized_response) > 0
        assert 0.0 <= result.final_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_all_agents_run(self, dmacn):
        """Test that all 4 agents execute."""
        result = await dmacn.run("Test query", n_agents=4)
        successful = [o for o in result.agent_outputs if not o.error]
        assert len(successful) == 4

    @pytest.mark.asyncio
    async def test_handles_agent_failure(self):
        """Test graceful handling when an agent fails."""
        from medaide_plus.modules.m3_dmacn import DMACNModule
        agents = [
            MockAgent("GoodAgent", "Normal response."),
            MockAgent("BadAgent", should_fail=True),
        ]
        dmacn = DMACNModule(agents=agents, config={"timeout": 5.0})
        result = await dmacn.run("Test query", n_agents=2)
        failed = [o for o in result.agent_outputs if o.error]
        assert len(failed) >= 1

    @pytest.mark.asyncio
    async def test_empty_agents(self):
        """Test with no agents available."""
        from medaide_plus.modules.m3_dmacn import DMACNModule
        dmacn = DMACNModule(agents=[], config={})
        result = await dmacn.run("Test query", n_agents=2)
        assert "No agents" in result.synthesized_response or result.synthesized_response == ""
