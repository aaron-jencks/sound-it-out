"""
Generic dataset preprocessing script for headered .bin format
Supports any HuggingFace dataset and custom tokenizers
"""
import os
import argparse
import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def write_datafile(filename, toks, token_dtype_bits):
    assert len(toks) < 2**31, "token count too large"
    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240520 # magic
    if token_dtype_bits == 16:
        header[1] = 1 # version
    else:
        header[1] = 2 # version (token dtype stored in header[3])
        header[3] = int(token_dtype_bits)
    header[2] = len(toks) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    if token_dtype_bits == 16:
        token_dtype = np.uint16
        maxtok = 2**16
    elif token_dtype_bits == 32:
        token_dtype = np.uint32
        maxtok = 2**32
    else:
        raise ValueError(f"Unsupported token dtype bits: {token_dtype_bits}")
    if not isinstance(toks, np.ndarray) or not toks.dtype == token_dtype:
        assert all(0 <= t < maxtok for t in toks), f"token dictionary too large for uint{token_dtype_bits}"
        toks_np = np.array(toks, dtype=token_dtype)
    else:
        toks_np = toks
    # write to file
    print(f"writing {len(toks):,} tokens to {filename}")
    with open(filename, "wb") as f:
        f.write(header.tobytes())
        f.write(toks_np.tobytes())

# ------------------------------------------

parser = argparse.ArgumentParser(description="Generic dataset preprocessing for headered .bin format")
parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset name (e.g., 'HuggingFaceFW/fineweb')")
parser.add_argument("--dataset_config", type=str, default=None, help="Dataset configuration/subset name (e.g., 'sample-10BT')")
parser.add_argument("--split", type=str, default="train", help="Dataset split to use (default: 'train')")
parser.add_argument("--text_field", type=str, default="text", help="Field name containing text data (default: 'text')")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for .bin files")
parser.add_argument("--shard_size", type=int, default=10**8, help="Size of each shard in tokens (default: 100M)")
parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to use: 'gpt2', 'cl100k_base', 'o200k_base', or path to custom tokenizer file")
parser.add_argument("--eot_token", type=int, default=None, help="End-of-text token ID (auto-detected if not specified)")
parser.add_argument("--token_dtype", type=int, choices=[16, 32], default=None, help="Token dtype bits (16 or 32). Auto-selects based on vocab if not set.")
parser.add_argument("--streaming", action="store_true", help="Use streaming dataset loading (IterableDataset).")
parser.add_argument("--shuffle_buffer", type=int, default=0, help="Shuffle buffer size for streaming datasets (0 disables).")
parser.add_argument("--shuffle_seed", type=int, default=1234, help="Seed for streaming shuffle.")
args = parser.parse_args()

# create the output directory if it doesn't exist
os.makedirs(args.output_dir, exist_ok=True)

# download/load dataset
print(f"Loading dataset: {args.dataset}")
if args.dataset_config:
    print(f"Config: {args.dataset_config}")
    dataset = load_dataset(args.dataset, name=args.dataset_config, split=args.split, streaming=args.streaming)
else:
    dataset = load_dataset(args.dataset, split=args.split, streaming=args.streaming)
print(f"Split: {args.split}")
if args.streaming:
    print("Streaming enabled")
    if args.shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=args.shuffle_buffer, seed=args.shuffle_seed)
        print(f"Streaming shuffle buffer: {args.shuffle_buffer}")
else:
    print(f"Size: {len(dataset)} documents")

# initialize the tokenizer
print(f"Loading tokenizer: {args.tokenizer}")
if args.tokenizer in ["gpt2", "cl100k_base", "o200k_base"]:
    enc = tiktoken.get_encoding(args.tokenizer)
    vocab_size = enc.n_vocab
    eot = enc._special_tokens['<|endoftext|>'] if args.eot_token is None else args.eot_token
    
    def tokenize(doc):
        tokens = [eot]
        tokens.extend(enc.encode_ordinary(doc[args.text_field]))
        tokens_np = np.array(tokens)
        assert (0 <= tokens_np).all() and (tokens_np < 2**token_dtype_bits).all(), f"token dictionary too large for uint{token_dtype_bits}"
        return tokens_np.astype(token_dtype_np)
    
else:
    # Load custom tokenizer from file
    try:
        from tokenizers import Tokenizer
        enc = Tokenizer.from_file(args.tokenizer)
        vocab_size = enc.get_vocab_size()
        print(f"Custom tokenizer loaded with vocab size: {vocab_size}")
        
        # Try to detect EOT token
        if args.eot_token is None:
            # Common EOT token names
            for token_name in ['<|endoftext|>', '<|eos|>', '[EOS]', '</s>', '<eos>', '<|bos|>', '<|pad|>']:
                eot = enc.token_to_id(token_name)
                if eot is not None:
                    print(f"Auto-detected EOT token: '{token_name}' (ID: {eot})")
                    break
            if eot is None:
                print("Warning: Could not auto-detect EOT token. Using ID 0. Specify with --eot_token if needed.")
                eot = 0
        else:
            eot = args.eot_token

        if eot is not None and eot >= vocab_size:
            print(
                f"WARNING: EOT token id {eot} is >= tokenizer vocab size {vocab_size}. "
                "This will create bin files containing out-of-vocab ids for the tokenizer; "
                "ensure your model vocab_size covers this id and avoid decoding it."
            )
        
        def tokenize(doc):
            tokens = [eot]
            encoded = enc.encode(doc[args.text_field])
            tokens.extend(encoded.ids)
            tokens_np = np.array(tokens)
            assert (0 <= tokens_np).all() and (tokens_np < 2**token_dtype_bits).all(), f"token dictionary too large for uint{token_dtype_bits}"
            return tokens_np.astype(token_dtype_np)
            
    except ImportError:
        print("Error: Custom tokenizer specified but 'tokenizers' library not installed.")
        print("Install with: pip install tokenizers")
        exit(1)
    except Exception as e:
        print(f"Error loading tokenizer from {args.tokenizer}: {e}")
        exit(1)

token_dtype_bits = args.token_dtype if args.token_dtype is not None else (32 if vocab_size > 2**16 - 1 else 16)
token_dtype_np = np.uint32 if token_dtype_bits == 32 else np.uint16
if eot is not None and eot >= 2**token_dtype_bits:
    raise ValueError(f"EOT token id {eot} does not fit in uint{token_dtype_bits}")

print(f"Using EOT token ID: {eot}")
print(f"Text field: '{args.text_field}'")
print(f"Shard size: {args.shard_size:,} tokens")
print(f"Output directory: {args.output_dir}")
print(f"Token dtype: uint{token_dtype_bits}")

# tokenize all documents and write output shards
nprocs = max(1, os.cpu_count() - 2)
print(f"Using {nprocs} processes for tokenization")

with mp.Pool(nprocs) as pool:
    shard_index = 0
    all_tokens_np = np.empty((args.shard_size,), dtype=token_dtype_np)
    token_count = 0
    progress_bar = None
    
    for tokens in pool.imap(tokenize, dataset, chunksize=16):
        if token_count + len(tokens) < args.shard_size:
            # simply append tokens to current shard
            all_tokens_np[token_count:token_count+len(tokens)] = tokens
            token_count += len(tokens)
            if progress_bar is None:
                progress_bar = tqdm(total=args.shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # write the current shard and start a new one
            split = "val" if shard_index == 0 else "train"
            filename = os.path.join(args.output_dir, f"data_{split}_{shard_index:06d}.bin")
            # split the document into whatever fits in this shard; the remainder goes to next one
            remainder = args.shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count:token_count+remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np, token_dtype_bits)
            shard_index += 1
            progress_bar = None
            # populate the next shard with the leftovers of the current doc
            all_tokens_np[0:len(tokens)-remainder] = tokens[remainder:]
            token_count = len(tokens)-remainder

    # write any remaining tokens as the last shard
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(args.output_dir, f"data_{split}_{shard_index:06d}.bin")
        write_datafile(filename, all_tokens_np[:token_count], token_dtype_bits)

print(f"\nPreprocessing complete! Created {shard_index + 1} shard(s) in {args.output_dir}")
