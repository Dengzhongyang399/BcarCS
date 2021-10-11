import javalang
import pandas as pd
from utils import get_sequence as func1
from utils import get_blocks_v1 as func2


class Pipeline:
    def __init__(self):
        self.sources = None
        self.blocks = None
        self.size = None

    def processSourceCode(self):
        def trans_to_sequences(ast):
            sequence = []
            func1(ast, sequence)
            return sequence

        def parse_program(func):
            tokens = javalang.tokenizer.tokenize(func)
            parser = javalang.parser.Parser(tokens)
            tree = parser.parse_member_declaration()
            return tree

        #PROCESS FILE

        source = pd.read_csv("data/test_source.csv", sep='\n', header=None,
                             encoding='utf-8')
        source.columns = ['code']
        source['code'] = source['code'].apply(parse_program)
        self.sources = source
        # source.to_pickle("source_code_ast.pkl")


        # READ FILE

        # source = pd.read_pickle("source_code_ast.pkl")
        self.sources = source
        print("finished...")

    def generate_block_seqs(self):

        def tree_to_seq(node):
            token = node.token
            result = [token]
            children = node.children
            for child in children:
                result.append(tree_to_seq(child))
            return result

        def tokens2seq(l, output):
            for tokens in l:
                if isinstance(tokens, list):
                    output.append("(")
                    tokens2seq(tokens, output)
                    output.append(")")
                else:
                    output.append(tokens)

        def trans2seq(r):
            train_api = open("data/test_staTree.txt", "a", encoding='utf-8')
            blocks = []
            func2(r, blocks)
            tokens = []
            result = []
            for b in blocks:
                btoken = tree_to_seq(b)
                tokens.append(btoken)
            for seq in tokens:
                seqResult = []
                tokens2seq(seq, seqResult)
                string = ""
                for word in seqResult:
                    string = string + word
                result.append(string)
            astSeq = ""
            for word in result:
                astSeq = astSeq + word + " "
            train_api.write(astSeq + "\n")
            print(astSeq)

        trees = pd.DataFrame(self.sources, copy=True)
        trees['code'].apply(trans2seq)
        print("finished...")

    def run(self):
        self.processSourceCode()
        self.generate_block_seqs()


ppl = Pipeline()
ppl.run()
