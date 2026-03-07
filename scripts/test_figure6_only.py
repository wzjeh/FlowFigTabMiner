import json
import logging
from src.assembly.evidence_assembler import EvidenceAssembler
from src.adjudication.data_synthesizer import DataSynthesizer

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

def test_fig6():
    fig_id = 'page_6_figure_0_t0'
    int_dir = 'data/intermediate/example/macro_cleaned'

    with open(f'{int_dir}/{fig_id}_evidence.json', 'r') as f:
        raw_evidence = json.load(f)

    assembler = EvidenceAssembler()

    # Pass the previous extraction data to assemble
    # assemble() returns a file path string
    file_path = assembler.assemble(fig_id, raw_evidence['raw_data'], int_dir)

    print('\n=======================================')
    if file_path:
        with open(file_path, 'r') as f:
            evidence_item = json.load(f)
            
        print('Final Assembled Caption:', evidence_item.get('meta', {}).get('caption'))
        print('Num Points:', len(evidence_item.get('raw_data', [])))
        
        synth = DataSynthesizer()
        # Mock globals
        global_vars = {}
        resolved_context = {}
        result = synth.synthesize(evidence_item, global_vars, resolved_context)
        
        print('\nSynthesis Result Length:', len(result))
        print(json.dumps(result[:2], indent=2, ensure_ascii=False))

if __name__ == '__main__':
    test_fig6()
