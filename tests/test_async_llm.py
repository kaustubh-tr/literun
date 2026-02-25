import os
import sys
import unittest

# Add the project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from literun import ChatOpenAI
from literun.events import MessageOutputStreamDelta, StreamEndEvent
from literun.prompt import PromptTemplate
from literun.usage import TokenUsage

from tests.helpers import FakeLLM

try:
    import openai  # noqa: F401

    HAS_OPENAI_SDK = True
except Exception:
    HAS_OPENAI_SDK = False


class TestAsyncBaseLLMBehavior(unittest.IsolatedAsyncioTestCase):
    async def test_fake_llm_agenerate_and_astream(self):
        llm = FakeLLM(
            model="fake-model",
            scripted_responses=[{"text": "ok", "tool_calls": [], "usage": None, "items": []}],
            scripted_streams=[[MessageOutputStreamDelta(id="1", delta="x"), StreamEndEvent(id="2")]],
        )

        resp = await llm.agenerate(messages=[{"role": "user", "content": "hi"}], stream=False)
        self.assertEqual(resp["text"], "ok")

        stream = await llm.agenerate(messages=[{"role": "user", "content": "hi"}], stream=True)
        adapter = llm.get_stream_adapter()
        chunks = []
        async for event in adapter.aprocess_stream(stream):
            chunks.append(event)
        self.assertEqual(len(chunks), 2)


@unittest.skipUnless(HAS_OPENAI_SDK, "openai sdk not installed")
class TestAsyncChatOpenAI(unittest.IsolatedAsyncioTestCase):
    async def test_aclose(self):
        llm = ChatOpenAI(model="gpt-5-nano", api_key="test-key")
        await llm.aclose()

    async def test_async_context_manager(self):
        async with ChatOpenAI(model="gpt-5-nano", api_key="test-key") as llm:
            self.assertIsInstance(llm, ChatOpenAI)

    async def test_prompt_normalization_still_sync_safe(self):
        llm = ChatOpenAI(model="gpt-5-nano", api_key="test-key")
        prompt = PromptTemplate().add_user("hello")
        normalized = llm.normalize_messages(prompt)
        self.assertTrue(isinstance(normalized, list))
        self.assertEqual(normalized[0]["role"], "user")
        await llm.aclose()


if __name__ == "__main__":
    unittest.main()
