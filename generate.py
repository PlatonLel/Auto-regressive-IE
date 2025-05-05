import torch
from tqdm import tqdm

class Generate:
    def __init__(self, model, loader, decoding_function=None):
        self.model = model
        self.loader = loader
        self.decoding_function = decoding_function
    
    def use(self):
        return self.generate(self.model, self.loader)
    
    @torch.no_grad()
    def generate(self, model, loader):

        model.eval()
        all_outs = []
        device = next(model.parameters()).device
        for x in tqdm(loader, desc="Processing text"):
            # Move input tensors to the device
            for k, v in x.items():
                if isinstance(v, torch.Tensor):
                    x[k] = v.to(device)
            if self.decoding_function is None:
                out = model.decode_batch(x)
            else:
                out = self.decoding_function(model, x)
            all_outs.extend(out)

        return all_outs
    
    def transform_data(self, all_true, all_outs, symetric=False, exclude_type=False):
        # extract entities and relations
        all_outs_ent = []
        all_outs_rel = []
        for i in all_outs:
            e, r = self.extract_entities_and_relations(i, symetric=symetric, exclude_type=exclude_type).values()
            all_outs_ent.append(e)
            all_outs_rel.append(r)

        return all_outs_ent, all_outs_rel
    
    @staticmethod
    def get_entities(output_seq):
        all_ents = []
        for i in output_seq:
            if len(i) == 3:
                s, e, lab = i
                if [lab, (s, e)] in all_ents:
                    continue
                all_ents.append([lab, (s, e)])
        return all_ents

    @staticmethod
    def get_relations(dec_i, symetric, exclude_type):
        relations = []

        if dec_i[-1] == "stop_entity":
            return relations

        index_end = dec_i.index("stop_entity")

        for i in range(index_end + 1, len(dec_i), 3):
            head, tail, r_type = dec_i[i:i + 3]

            if exclude_type:
                head = head[0], head[1]
                tail = tail[0], tail[1]

            if symetric or r_type in ["COMPARE", "CONJUNCTION"]:  # sort the head and tail by start index
                if head[0] > tail[0]:
                    head, tail = tail, head

            if head != tail and [r_type, (head, tail)] not in relations:
                relations.append([r_type, (head, tail)])

        return relations
    

    def extract_entities_and_relations(self, input_seq, symetric, exclude_type):
        try:
            relations_triples = self.get_relations(input_seq, symetric, exclude_type)
        except:
            relations_triples = []
        entities = self.get_entities(input_seq)

        return {
            "entities": entities,
            "relations_triples": relations_triples
        }

