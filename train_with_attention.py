import torch
from torch import nn
import pandas as pd
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import copy
import random
import wandb
import argparse

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Computation device: {device}")

# Special tokens
START_TOKEN, END_TOKEN, PAD_TOKEN = '<', '>', '_'
TEACHER_FORCING_RATIO = 0.5

# --- Utility Functions (from your code) ---

def add_padding(source_data, MAX_LENGTH):
    padded_source_strings = []
    for s in source_data:
        s = START_TOKEN + s + END_TOKEN
        s = s[:MAX_LENGTH]
        s += PAD_TOKEN * (MAX_LENGTH - len(s))
        padded_source_strings.append(s)
    return padded_source_strings

def generate_string_to_sequence(source_data, source_char_index_dict):
    source_sequences = []
    for s in source_data:
        source_sequences.append(get_chars(s, source_char_index_dict))
    return pad_sequence(source_sequences, batch_first=True, padding_value=2)

def get_chars(string, char_index_dict):
    return torch.tensor([char_index_dict[c] for c in string], device=device)

def preprocess_data(source_data, target_data):
    data = {
        "source_chars": [START_TOKEN, END_TOKEN, PAD_TOKEN],
        "target_chars": [START_TOKEN, END_TOKEN, PAD_TOKEN],
        "source_char_index": {START_TOKEN: 0, END_TOKEN: 1, PAD_TOKEN: 2},
        "source_index_char": {0: START_TOKEN, 1: END_TOKEN, 2: PAD_TOKEN},
        "target_char_index": {START_TOKEN: 0, END_TOKEN: 1, PAD_TOKEN: 2},
        "target_index_char": {0: START_TOKEN, 1: END_TOKEN, 2: PAD_TOKEN},
        "source_len": 3,
        "target_len": 3,
        "source_data": source_data,
        "target_data": target_data,
        "source_data_seq": [],
        "target_data_seq": []
    }
    data["INPUT_MAX_LENGTH"] = max(len(s) for s in source_data) + 2
    data["OUTPUT_MAX_LENGTH"] = max(len(s) for s in target_data) + 2

    padded_source = add_padding(source_data, data["INPUT_MAX_LENGTH"])
    padded_target = add_padding(target_data, data["OUTPUT_MAX_LENGTH"])

    for i in range(len(padded_source)):
        for c in padded_source[i]:
            if c not in data["source_char_index"]:
                data["source_chars"].append(c)
                idx = len(data["source_chars"]) - 1
                data["source_char_index"][c] = idx
                data["source_index_char"][idx] = c
        for c in padded_target[i]:
            if c not in data["target_char_index"]:
                data["target_chars"].append(c)
                idx = len(data["target_chars"]) - 1
                data["target_char_index"][c] = idx
                data["target_index_char"][idx] = c

    data['source_data_seq'] = generate_string_to_sequence(padded_source, data['source_char_index'])
    data['target_data_seq'] = generate_string_to_sequence(padded_target, data['target_char_index'])
    data["source_len"] = len(data["source_chars"])
    data["target_len"] = len(data["target_chars"])
    return data

def get_cell_type(cell_type):
    if cell_type == "RNN":
        return nn.RNN
    elif cell_type == "LSTM":
        return nn.LSTM
    elif cell_type == "GRU":
        return nn.GRU
    else:
        raise ValueError("Specify correct cell type")

# --- Model Classes (from your code) ---

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze().unsqueeze(1)
        weights = F.softmax(scores, dim=0)
        weights = weights.permute(2,1,0)
        keys = keys.permute(1,0,2)
        context = torch.bmm(weights, keys)
        return context, weights

class Encoder(nn.Module):
    def __init__(self, h_params, data, device):
        super().__init__()
        self.embedding = nn.Embedding(data["source_len"], h_params["char_embd_dim"])
        self.cell = get_cell_type(h_params["cell_type"])(h_params["char_embd_dim"], h_params["hidden_layer_neurons"],
                                                         num_layers=h_params["number_of_layers"], batch_first=True)
        self.device = device
        self.h_params = h_params
        self.data = data

    def forward(self, input, encoder_curr_state):
        input_length = self.data["INPUT_MAX_LENGTH"]
        batch_size = self.h_params["batch_size"]
        hidden_neurons = self.h_params["hidden_layer_neurons"]
        layers = self.h_params["number_of_layers"]
        encoder_states = torch.zeros(input_length, layers, batch_size, hidden_neurons, device=self.device)
        for i in range(input_length):
            current_input = input[:, i].view(batch_size, 1)
            _, encoder_curr_state = self.forward_step(current_input, encoder_curr_state)
            if self.h_params["cell_type"] == "LSTM":
                encoder_states[i] = encoder_curr_state[1]
            else:
                encoder_states[i] = encoder_curr_state
        return encoder_states, encoder_curr_state

    def forward_step(self, current_input, prev_state):
        embd_input = self.embedding(current_input)
        output, prev_state = self.cell(embd_input, prev_state)
        return output, prev_state

    def getInitialState(self):
        return torch.zeros(self.h_params["number_of_layers"], self.h_params["batch_size"],
                           self.h_params["hidden_layer_neurons"], device=self.device)

class Decoder(nn.Module):
    def __init__(self, h_params, data, device):
        super().__init__()
        self.attention = Attention(h_params["hidden_layer_neurons"]).to(device)
        self.embedding = nn.Embedding(data["target_len"], h_params["char_embd_dim"])
        self.cell = get_cell_type(h_params["cell_type"])(h_params["hidden_layer_neurons"] + h_params["char_embd_dim"],
                                                         h_params["hidden_layer_neurons"], num_layers=h_params["number_of_layers"], batch_first=True)
        self.fc = nn.Linear(h_params["hidden_layer_neurons"], data["target_len"])
        self.softmax = nn.LogSoftmax(dim=2)
        self.h_params = h_params
        self.data = data
        self.device = device

    def forward(self, decoder_current_state, encoder_final_layers, target_batch, loss_fn, teacher_forcing_enabled=True):
        batch_size = self.h_params["batch_size"]
        decoder_current_input = torch.full((batch_size, 1), self.data["target_char_index"][START_TOKEN], device=self.device)
        embd_input = self.embedding(decoder_current_input)
        curr_embd = F.relu(embd_input)
        decoder_actual_output = []
        attentions = []
        loss = 0
        use_teacher_forcing = teacher_forcing_enabled and (random.random() < TEACHER_FORCING_RATIO)
        for i in range(self.data["OUTPUT_MAX_LENGTH"]):
            decoder_output, decoder_current_state, attn_weights = self.forward_step(decoder_current_input, decoder_current_state, encoder_final_layers)
            attentions.append(attn_weights)
            topv, topi = decoder_output.topk(1)
            decoder_current_input = topi.squeeze().detach()
            decoder_actual_output.append(decoder_current_input)
            if target_batch is not None:
                curr_target_chars = target_batch[:, i]
                if i < self.data["OUTPUT_MAX_LENGTH"] - 1:
                    if use_teacher_forcing:
                        decoder_current_input = target_batch[:, i + 1].view(self.h_params["batch_size"], 1)
                    else:
                        decoder_current_input = decoder_current_input.view(self.h_params["batch_size"], 1)
                decoder_output = decoder_output[:, -1, :]
                loss += loss_fn(decoder_output, curr_target_chars)
        decoder_actual_output = torch.cat(decoder_actual_output, dim=0).view(self.data["OUTPUT_MAX_LENGTH"], self.h_params["batch_size"]).transpose(0, 1)
        correct = (decoder_actual_output == target_batch).all(dim=1).sum().item()
        return decoder_actual_output, attentions, loss, correct

    def forward_step(self, current_input, prev_state, encoder_final_layers):
        embd_input = self.embedding(current_input)
        if self.h_params["cell_type"] == "LSTM":
            context, attn_weights = self.attention(prev_state[1][-1, :, :], encoder_final_layers)
        else:
            context, attn_weights = self.attention(prev_state[-1, :, :], encoder_final_layers)
        curr_embd = F.relu(embd_input)
        input_gru = torch.cat((curr_embd, context), dim=2)
        output, prev_state = self.cell(input_gru, prev_state)
        output = self.softmax(self.fc(output))
        return output, prev_state, attn_weights

class MyDataset(Dataset):
    def __init__(self, data):
        self.source_data_seq = data[0]
        self.target_data_seq = data[1]
    def __len__(self):
        return len(self.source_data_seq)
    def __getitem__(self, idx):
        return self.source_data_seq[idx], self.target_data_seq[idx]

def evaluate(encoder, decoder, data, dataloader, device, h_params, loss_fn, use_teacher_forcing=False):
    correct_predictions = 0
    total_loss = 0
    total_predictions = len(dataloader.dataset)
    number_of_batches = len(dataloader)
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        for batch_num, (source_batch, target_batch) in enumerate(dataloader):
            encoder_initial_state = encoder.getInitialState()
            if h_params["cell_type"] == "LSTM":
                encoder_initial_state = (encoder_initial_state, encoder.getInitialState())
            encoder_states, encoder_final_state = encoder(source_batch, encoder_initial_state)
            decoder_current_state = encoder_final_state
            encoder_final_layer_states = encoder_states[:, -1, :, :]
            decoder_output, attentions, loss, correct = decoder(decoder_current_state, encoder_final_layer_states, target_batch, loss_fn, use_teacher_forcing)
            correct_predictions += correct
            total_loss += loss
    accuracy = correct_predictions / total_predictions
    total_loss /= number_of_batches
    return accuracy, total_loss

def train_loop(encoder, decoder, h_params, data, data_loader, device, val_dataloader, use_teacher_forcing=True):
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=h_params["learning_rate"])
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=h_params["learning_rate"])
    loss_fn = nn.NLLLoss()
    total_predictions = len(data_loader.dataset)
    total_batches = len(data_loader)
    for ep in range(h_params["epochs"]):
        total_correct = 0
        total_loss = 0
        encoder.train()
        decoder.train()
        for batch_num, (source_batch, target_batch) in enumerate(data_loader):
            encoder_initial_state = encoder.getInitialState()
            if h_params["cell_type"] == "LSTM":
                encoder_initial_state = (encoder_initial_state, encoder.getInitialState())
            encoder_states, encoder_final_state = encoder(source_batch, encoder_initial_state)
            decoder_current_state = encoder_final_state
            encoder_final_layer_states = encoder_states[:, -1, :, :]
            decoder_output, attentions, loss, correct = decoder(decoder_current_state, encoder_final_layer_states, target_batch, loss_fn, use_teacher_forcing)
            total_correct += correct
            total_loss += loss.item() / data["OUTPUT_MAX_LENGTH"]
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            loss.backward()
            encoder_optimizer.step()
            decoder_optimizer.step()
        train_acc = total_correct / total_predictions
        train_loss = total_loss / total_batches
        val_acc, val_loss = evaluate(encoder, decoder, data, val_dataloader, device, h_params, loss_fn, False)
        print(f"ep: {ep} train acc: {train_acc} train loss: {train_loss} val acc: {val_acc} val loss: {val_loss}")
        wandb.log({"train_accuracy": train_acc, "train_loss": train_loss, "val_accuracy": val_acc, "val_loss": val_loss, "epoch": ep})
    return loss_fn

def prepare_dataloaders(train_source, train_target, val_source, val_target, h_params):
    data = preprocess_data(copy.copy(train_source), copy.copy(train_target))
    training_data = [data["source_data_seq"], data['target_data_seq']]
    train_dataset = MyDataset(training_data)
    train_dataloader = DataLoader(train_dataset, batch_size=h_params["batch_size"], shuffle=True)
    val_padded_source_strings = add_padding(val_source, data["INPUT_MAX_LENGTH"])
    val_padded_target_strings = add_padding(val_target, data["OUTPUT_MAX_LENGTH"])
    val_source_sequences = generate_string_to_sequence(val_padded_source_strings, data['source_char_index'])
    val_target_sequences = generate_string_to_sequence(val_padded_target_strings, data['target_char_index'])
    validation_data = [val_source_sequences, val_target_sequences]
    val_dataset = MyDataset(validation_data)
    val_dataloader = DataLoader(val_dataset, batch_size=h_params["batch_size"], shuffle=True)
    return train_dataloader, val_dataloader, data

def train(h_params, data, device, data_loader, val_dataloader, use_teacher_forcing=True):
    encoder = Encoder(h_params, data, device).to(device)
    decoder = Decoder(h_params, data, device).to(device)
    train_loop(encoder, decoder, h_params, data, data_loader, device, val_dataloader, use_teacher_forcing)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Train a seq2seq model with specified hyperparameters")
    parser.add_argument("-wp", "--wandb_project", type=str, default="DL proj", help="Weights & Biases project name")
    parser.add_argument("-e", "--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.0001, help="Learning rate")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-embd_dim", "--char_embd_dim", type=int, default=256, help="Char embedding dimension")
    parser.add_argument("-hid_neur", "--hidden_layer_neurons", type=int, default=256, help="Hidden layer neurons")
    parser.add_argument("-num_layers", "--number_of_layers", type=int, default=3, help="Number of layers")
    parser.add_argument("-cell", "--cell_type", choices=["RNN", "LSTM", "GRU"], default="LSTM", help="RNN cell type")
    parser.add_argument("-do", "--dropout", type=float, default=0, help="Dropout probability")
    parser.add_argument("-opt", "--optimizer", choices=["adam", "nadam"], default="adam", help="Optimizer")
    parser.add_argument("-train_path", "--train_path", type=str, required=True, help="Path to training data CSV")
    parser.add_argument("-test_path", "--test_path", type=str, required=True, help="Path to test data CSV")
    parser.add_argument("-val_path", "--val_path", type=str, required=True, help="Path to validation data CSV")
    args = parser.parse_args()
    return args

def main():
    wandb.login()
    args = parse_arguments()
    h_params = {
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "char_embd_dim": args.char_embd_dim,
        "hidden_layer_neurons": args.hidden_layer_neurons,
        "number_of_layers": args.number_of_layers,
        "cell_type": args.cell_type,
        "dropout": args.dropout,
        "optimizer": args.optimizer
    }
    train_csv = args.train_path
    test_csv = args.test_path
    val_csv = args.val_path
    train_df = pd.read_csv(train_csv, header=None)
    train_source, train_target = train_df[0].to_numpy(), train_df[1].to_numpy()
    test_df = pd.read_csv(test_csv, header=None)
    val_df = pd.read_csv(val_csv, header=None)
    val_source, val_target = val_df[0].to_numpy(), val_df[1].to_numpy()
    config = h_params
    run = wandb.init(project=args.wandb_project, name=f"{config['cell_type']}_{config['optimizer']}_ep_{config['epochs']}_lr_{config['learning_rate']}_embd_{config['char_embd_dim']}_hid_lyr_neur_{config['hidden_layer_neurons']}_bs_{config['batch_size']}_enc_layers_{config['number_of_layers']}_dec_layers_{config['number_of_layers']}_dropout_{config['dropout']}", config=config)
    train_dataloader, val_dataloader, data = prepare_dataloaders(train_source, train_target, val_source, val_target, h_params)
    train(h_params, data, device, train_dataloader, val_dataloader)

if __name__ == "__main__":
    main()
