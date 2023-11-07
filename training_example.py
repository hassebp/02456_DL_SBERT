import datasets
from transformers import BertTokenizer, BertModel
import torch
from tqdm.auto import tqdm
from transformers.optimization import get_linear_schedule_with_warmup
import os
import numpy




def mean_pool(token_embeds, attention_mask):
    # reshape attention_mask to cover 768-dimension embeddings
    in_mask = attention_mask.unsqueeze(-1).expand(
        token_embeds.size()
    ).float()
    # perform mean-pooling but exclude padding tokens (specified by in_mask)
    pool = torch.sum(token_embeds * in_mask, 1) / torch.clamp(
        in_mask.sum(1), min=1e-9
    )
    return pool



    
    
    
    
def test_snli():
    snli = datasets.load_dataset('snli', split='train')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = snli.filter(
        lambda x: 0 if x['label'] == -1 else 1
    )
    
    print(type(dataset))
    p.p
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased').to(device)  
    
    
    
        
    all_cols = ['label']

    for part in ['premise', 'hypothesis']:
        dataset = dataset.map(
            lambda x: tokenizer(
                x[part], max_length=128, padding='max_length',
                truncation=True
            ), batched=True
        )
        for col in ['input_ids', 'attention_mask']:
            dataset = dataset.rename_column(
                col, part+'_'+col
            )
            all_cols.append(part+'_'+col)

    # covert dataset features to PyTorch tensors
    dataset.set_format(type='torch', columns=all_cols)
    
    # initialize the dataloader
    batch_size = 16
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # we would initialize everything first
    optim = torch.optim.Adam(model.parameters(), lr=2e-5)
    # and setup a warmup for the first ~10% steps
    total_steps = int(len(dataset) / batch_size)
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
            optim, num_warmup_steps=warmup_steps,
        num_training_steps=total_steps - warmup_steps
    )
    loss_func = torch.nn.CrossEntropyLoss()
    ffnn = torch.nn.Linear(768*3, 3)
    for epoch in range(1):
        model.train()  # make sure model is in training mode
        # initialize the dataloader loop with tqdm (tqdm == progress bar)
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # zero all gradients on each new step
            optim.zero_grad()
            # prepare batches and more all to the active device
            inputs_ids_a = batch['premise_input_ids'].to(device)
            inputs_ids_b = batch['hypothesis_input_ids'].to(device)
            attention_a = batch['premise_attention_mask'].to(device)
            attention_b = batch['hypothesis_attention_mask'].to(device)
            label = batch['label'].to(device)
            # extract token embeddings from BERT
            u = model(
                inputs_ids_a, attention_mask=attention_a
            )[0]  # all token embeddings A
            v = model(
                inputs_ids_b, attention_mask=attention_b
            )[0]  # all token embeddings B
            # get the mean pooled vectors
            u = mean_pool(u, attention_a)
            v = mean_pool(v, attention_b)
            # build the |u-v| tensor
            uv = torch.sub(u, v)
            uv_abs = torch.abs(uv)
            
            # Move tensors to the same device
            u, v, uv_abs = u.to(device), v.to(device), uv_abs.to(device)
            # concatenate u, v, |u-v|
            x = torch.cat([u, v, uv_abs], dim=-1)
            # process concatenated tensor through FFNN
            x = ffnn(x.to(device))
            # calculate the 'softmax-loss' between predicted and true label
            loss = loss_func(x, label)
            # using loss, calculate gradients and then optimize
            loss.backward()
            optim.step()
            # update learning rate scheduler
            scheduler.step()
            # update the TDQM progress bar
            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

    
    model_path = './sbert_test_a'

    if not os.path.exists(model_path):
        os.mkdir(model_path)

    model.save_pretrained(model_path)