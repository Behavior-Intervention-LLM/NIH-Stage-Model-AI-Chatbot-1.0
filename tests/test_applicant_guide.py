import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import uuid

from app.agents.grant_partner_agent import GrantPartnerAgent
from app.agents.rag_agent import RAGAgent
from app.core.orchestrator import Orchestrator
from app.core.types import AgentOutput, Artifact, Citation, SessionState, ToolResult
from app.tools.applicant_guide import ApplicantGuideTool, ensure_guide_index, _program_sections
from app.tools.base import BaseTool, ToolRegistry
from app.tools.document_loader import DocumentChunk
from app.tools.vector_store import SimpleVectorStore


class GuideIngestionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.pdf = self.root / 'guide.pdf'
        self.pdf.write_bytes(b'first edition')
        self.cache = self.root / 'cache'

    def extract(self, path):
        return [DocumentChunk(path.read_text(), path.name, 0, {'page': 4})], {'page_count': 1}

    def test_unchanged_pdf_is_reused_and_changed_pdf_replaces_evidence(self):
        with patch('app.tools.applicant_guide.extract_guide', side_effect=self.extract) as extract:
            first = ensure_guide_index(self.pdf, self.cache)
            self.assertEqual(ensure_guide_index(self.pdf, self.cache), first)
            self.assertEqual(extract.call_count, 1)
            self.pdf.write_bytes(b'revised policy')
            second = ensure_guide_index(self.pdf, self.cache)
            self.assertNotEqual(first, second)
            store = SimpleVectorStore(second)
            self.assertEqual(len(store.chunks), 1)
            self.assertEqual(store.search('first'), [])
            self.assertTrue(store.search('revised'))
            self.assertEqual(json.loads((second / 'manifest.json').read_text())['chunk_count'], 1)

    def test_failed_update_does_not_publish_partial_index_or_use_old_guide(self):
        with patch('app.tools.applicant_guide.extract_guide', side_effect=self.extract):
            tool = ApplicantGuideTool(self.pdf, self.cache)
            self.assertTrue(tool.run('first').citations)
        self.pdf.write_bytes(b'broken update')
        with patch('app.tools.applicant_guide.extract_guide', side_effect=ValueError('broken PDF')):
            result = tool.run('first')
        self.assertFalse(result.success)
        self.assertEqual(result.citations, [])
        self.assertEqual(len(list(self.cache.glob('*/manifest.json'))), 1)

    def test_missing_pdf_returns_explicit_error(self):
        result = ApplicantGuideTool(self.root / 'missing.pdf', self.cache).run('aims')
        self.assertFalse(result.success)
        self.assertTrue(result.error)
        self.assertEqual(result.citations, [])

    def test_program_filter_keeps_research_instructions_for_r01(self):
        def extract(path):
            return [DocumentChunk('Specific Aims R01 application', path.name, i,
                                  {'page': i + 1, 'section': section})
                    for i, section in enumerate(['G.410 - Career', 'G.430 - Fellowship', 'G.400 - Research'])], {}
        # Make content unique so the store retains each program's passage.
        def unique_extract(path):
            chunks, edition = extract(path)
            for c in chunks:
                c.content += ' ' + c.metadata['section']
            return chunks, edition
        with patch('app.tools.applicant_guide.extract_guide', side_effect=unique_extract):
            result = ApplicantGuideTool(self.pdf, self.cache).run('R01 Specific Aims')
        self.assertEqual(len(result.citations), 1)
        self.assertEqual(result.citations[0].metadata['section'], 'G.400 - Research')
        self.assertIn('G.420', _program_sections('K12 application'))
        self.assertIsNone(_program_sections('Compare R01 and F31 requirements'))
        self.assertIsNone(_program_sections('Compare R01 and X99 requirements'))

    def test_page_extraction_keeps_words_and_metadata(self):
        class Page:
            def __init__(self, text):
                self.text = text
            def extract_text(self, **kwargs):
                self.assertion = kwargs['x_tolerance']
                return self.text
            def flush_cache(self):
                pass
        pages = [Page('FORMS VERSION I SERIES\nReleased: December, 2025'),
                 Page('G.400 - Research Plan\nSpecific Aims are required.'),
                 Page('Research Strategy conditions continue here.')]
        from app.tools.applicant_guide import extract_guide
        with patch('app.tools.applicant_guide.pdfplumber.open') as open_pdf:
            open_pdf.return_value.__enter__.return_value.pages = pages
            chunks, edition = extract_guide(self.pdf)
        self.assertEqual(edition['forms_version'], 'I')
        self.assertEqual(chunks[-1].metadata['page'], 3)
        self.assertEqual(chunks[-1].metadata['section'], 'G.400 - Research Plan')
        self.assertIn('Specific Aims', chunks[1].content)
        self.assertTrue(all(p.assertion == 1 for p in pages))


class FakeGuide(BaseTool):
    def __init__(self):
        super().__init__('applicant_guide_tool', 'test')
        self.queries = []

    def run(self, query, **kwargs):
        self.queries.append(query)
        return ToolResult(text='guide', citations=[Citation(
            source='guide.pdf', passage='Specific Aims are required. ' + 'Context. ' * 100 + 'Conditional requirement at the end.',
            relevance_score=.8,
            metadata={'semantic_score': .8, 'term_coverage': .9, 'page': 146,
                      'section': 'G.400', 'forms_version': 'I', 'released': 'December, 2025'},
        )])


class GuideWorkflowTests(unittest.TestCase):
    def state(self):
        return SessionState(session_id=str(uuid.uuid4()))

    def test_partner_prompt_contains_full_passage_and_rejects_weak_matches(self):
        state = self.state()
        good = FakeGuide().run('aims').citations[0]
        bad = Citation(source='weak.pdf', passage='unrelated', relevance_score=.01,
                       metadata={'semantic_score': .01, 'term_coverage': .1})
        state.artifacts = [Artifact(tool_name='applicant_guide_tool', result_type='text',
                                    content='truncated summary', citations=[good, bad])]
        system, prompt, citations, cap = GrantPartnerAgent()._build_llm_prompts(state, 'Draft my aims')
        self.assertIn('Conditional requirement at the end.', prompt)
        self.assertIn('PDF p. 146', prompt)
        self.assertNotIn('unrelated', prompt)
        self.assertIn('NOFO', system)
        self.assertIn('Inferred stage: None', prompt)
        self.assertEqual(citations, [good])

    def test_application_graph_routes_and_streams_without_stage_classification(self):
        for message, override in [('What are the Specific Aims requirements?', None),
                                  ('Check this budget.', 'grant_partner')]:
            with self.subTest(message=message):
                guide = FakeGuide()
                registry = ToolRegistry()
                registry.register(guide)
                orch = Orchestrator(registry)
                intent = AgentOutput(decision={'workflow': 'navigator', 'need_stage': True,
                                               'intent_label': 'general_qa', 'query_type': 'general_qa'})
                tokens = []
                with patch.object(orch.agents['intent_agent'], 'run', return_value=intent), \
                     patch.object(orch.agents['stage_agent'], 'run', side_effect=AssertionError('stage should be skipped')), \
                     patch('chat_history.load_attachments', return_value=[]), \
                     patch('app.agents.responder_agent.llm_client.is_enabled', return_value=True), \
                     patch('app.agents.responder_agent.llm_client.chat_text_stream', return_value=iter(['Guide answer'])):
                    result = orch._graph.invoke({'session_id': str(uuid.uuid4()), 'user_message': message,
                                                'workflow_override': override, 'stream_handler': tokens.append})
                self.assertEqual(result['reply'], 'Guide answer')
                self.assertEqual(tokens, ['Guide answer'])
                self.assertIn('grant_partner_agent', result['called_agents'])
                self.assertNotIn('stage_agent', result['called_agents'])
                self.assertEqual(result['last_output'].metadata['citations'][0]['metadata']['page'], 146)
                self.assertTrue(guide.queries)

    def test_normal_retrieval_and_accepted_application_offer(self):
        state = self.state()
        state.slots.stage = 'II'
        out = RAGAgent().plan(state, 'Explain efficacy')
        self.assertEqual(out.actions[0].tool_name, 'versioned_rag_tool')
        state.slots.extracted_features.update(workflow='grant_partner', accepted_offer='Draft Specific Aims')
        out = RAGAgent().plan(state, 'Yes please')
        self.assertEqual(out.actions[0].tool_name, 'applicant_guide_tool')
        self.assertEqual(out.actions[0].tool_args['query'], 'Draft Specific Aims')


if __name__ == '__main__':
    unittest.main()
