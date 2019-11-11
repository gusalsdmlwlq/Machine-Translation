import argparse


class Config:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--max_epochs", default=10, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--warmup_steps", default=8000, type=int)
    parser.add_argument("--d_model", default=512, type=int)
    parser.add_argument("--d_ff", default=2048, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--num_layers", default=6, type=int)
    parser.add_argument("--max_len", default=150, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--no_cuda", action="store_true", default=False)
    parser.add_argument("--print_steps", default=1000, type=int)
    parser.add_argument("--validation_rate", default=0.05, type=float)
    parser.add_argument("--vocab_size", default=32000, type=int)