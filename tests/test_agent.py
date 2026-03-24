"""Tests for agent memory and candidate tracking."""

from peptagent.agent.memory import AgentMemory


class TestAgentMemory:
    def test_add_and_retrieve_candidate(self):
        mem = AgentMemory()
        mem.add_candidate(
            sequence="ACDEFGHIK",
            properties={"hemolysis": {"probability": 0.1}},
            reliability_score=0.85,
        )
        assert mem.was_evaluated("ACDEFGHIK")
        assert not mem.was_evaluated("XXXXXXXXX")

    def test_update_existing_candidate(self):
        mem = AgentMemory()
        mem.add_candidate("ACDEF", properties={"hemolysis": {"prob": 0.1}})
        mem.add_candidate("ACDEF", properties={"toxicity": {"prob": 0.2}}, reliability_score=0.9)

        record = mem.candidates["ACDEF"]
        assert "hemolysis" in record.properties
        assert "toxicity" in record.properties
        assert record.reliability_score == 0.9

    def test_top_candidates(self):
        mem = AgentMemory()
        mem.add_candidate("AAA", reliability_score=0.9)
        mem.add_candidate("BBB", reliability_score=0.7)
        mem.add_candidate("CCC", reliability_score=0.95)
        mem.add_candidate("DDD", reliability_score=0.3)

        top = mem.get_top_candidates(n=2)
        assert len(top) == 2
        assert top[0].sequence == "CCC"
        assert top[1].sequence == "AAA"

    def test_top_candidates_with_min_confidence(self):
        mem = AgentMemory()
        mem.add_candidate("AAA", reliability_score=0.9)
        mem.add_candidate("BBB", reliability_score=0.3)

        top = mem.get_top_candidates(n=5, min_confidence=0.5)
        assert len(top) == 1
        assert top[0].sequence == "AAA"

    def test_message_tracking(self):
        mem = AgentMemory()
        mem.add_message("system", "hello")
        mem.add_message("user", "design a peptide")
        assert len(mem.messages) == 2

    def test_iteration_tracking(self):
        mem = AgentMemory()
        assert mem.iteration == 0
        mem.advance_iteration()
        assert mem.iteration == 1
