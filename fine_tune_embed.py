import transformers
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from scipy.spatial.distance import cosine
from collections import defaultdict



def read_model(model_path):
  model = transformers.BertModel.from_pretrained('bert-base-uncased')
  model.load_state_dict(torch.load(model_path))
  tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
  return model,tokenizer
  
  
def handle_tokenize(texts, tokenizer, labels=None):  
    from torch.utils.data import TensorDataset
    import torch
    print('starting tokenizing...')
    encoding = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)  
    # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
    ids = encoding['input_ids'] 
    mask = encoding['attention_mask']
    token_type_ids = encoding['token_type_ids']

    if labels:
        targets = torch.tensor(labels)
        print('seq, mask and labels are ready')
        return TensorDataset(ids, mask, token_type_ids, targets)
    else:
        print('seq, mask are ready')
        return TensorDataset(ids, mask, token_type_ids)
      

      
  def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
    sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    return sum_embeddings / sum_mask

def dense(features):
    m = nn.Tanh()
    linear = nn.Linear(features.shape[1], 256)
    return  m(linear(features))


def infer_bert_model(testing_loader, model):
    
    model.eval()
    test_embeddings = []
    test_targets = []
    with torch.no_grad():
        for step, batch in enumerate(testing_loader):
            print(step)
            batch = [r.to(model.device) for r in batch]
            if len(batch) == 4:
                ids, mask, token_type_ids, label = batch
                labels_in_batch = True
            elif len(batch) == 3:
                ids, mask, token_type_ids = batch
                labels_in_batch = False
            model_output = model(ids, mask, token_type_ids)
            sentence_embeddings = mean_pooling(model_output, mask)
            sentence_embeddings = dense(sentence_embeddings)
            test_embeddings.append(sentence_embeddings.cpu().detach().numpy())
            if labels_in_batch:
                test_targets.append(label.cpu().detach().numpy())

    test_embeddings = np.concatenate(test_embeddings, axis=0)
    if labels_in_batch:
        test_targets = np.concatenate(test_targets, axis=0)
        return test_targets, test_embeddings
    return test_embeddings
  
  
 def mds_embedding(test_embedding):
  dists=cosine_distances(test_embedding)
  mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=60, max_iter=90000)
  test_embedding_transform = mds.fit(dists)
  return test_embedding_transform



if __name__=="__main__":
  
  model_path ="/content/drive/MyDrive/phd_docs/twitter_mul_lab_finetuned.model"
  model,tokenizer = read_model(model_path)
  test_set = handle_tokenize(df["prep_text"].to_list(),tokenizer)

  VALID_BATCH_SIZE = 50
  testing_loader = DataLoader(test_set, batch_size=VALID_BATCH_SIZE, shuffle=False, num_workers=4)
  test_embeddings=infer_bert_model(testing_loader, model)


  fine_tune_embedding=mds_embedding(test_embeddings)
  df["X_0_fine_tune"] = fine_tune_embedding.embedding_[:,0]
  df["X_1_fine_tune"] = fine_tune_embedding.embedding_[:,1]

  fig = px.scatter(df, x="X_0_fine_tune", y="X_1_fine_tune", color="MV", opacity=0.8,hover_data=["prep_text","MV_count"])
  fig.show()
  fig.write_image("MV_coloring_fine_tune.pdf")
        
        
  
