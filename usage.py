import logging
import torch

from generate import Generate
from preprocess import GraphIEData

# logging level
logging.basicConfig(level=logging.INFO)

def usage(data, model):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        import flair

        flair.device = torch.device('cpu')
        device = torch.device("cpu")
        
    data = [{"tokens": list(data.split()), "ner": [], "rel": [], "seq": []}]
    d = GraphIEData(data, type='use')
    
    model.to(device)
    loader = model.create_dataloader(d, batch_size=1, shuffle=False)
    model.eval()

    generator = Generate(model, loader)
    return generator.use()

# if __name__=="__main__":
#     print(usage('Wonder Woman is a superheroine appearing in American comic books published by DC Comics. The character is a founding member of the Justice League. The character first appeared in "All Star Comics" #8 published October 21, 1941 with her first feature in "Sensation Comics" #1 in January 1942. The "Wonder Woman" title has been published by DC Comics almost continuously ever since. In her homeland, the island nation of Themyscira, her official title is Princess Diana of Themyscira. When blending into the society outside of her homeland, she sometimes adopts her civilian identity Diana Prince'))
