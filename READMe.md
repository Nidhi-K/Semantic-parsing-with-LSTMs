To run original model with Attention: `python3 main.py` <br/>
To run with linear scheduled sampling: `python3 main_ss_linear.py` <br/>
To run with inverse sigmoid scheduled sampling: `python3 main_ss_siginv.py` <br/>
Optional Arguments with `main.py`: <br/>
  `-h, --help`: show this help message and exit <br/>
  `--do_nearest_neighbor`:
                        run the nearest neighbor model <br/>
  `--test_output_path TEST_OUTPUT_PATH`:
                        path to write blind test results <br/>
  `--epochs EPOCHS`:       num epochs to train for <br/>
  `--lr LR`:
                        learning rate (step size) for optimizer <br/>
  `--batch_size BATCH_SIZE`:
                        batch size <br/>
  `--decoder_len_limit DECODER_LEN_LIMIT`:
                        output length limit of the decoder <br/>
  `--input_dim INPUT_DIM`:
                        input vector dimensionality <br/>
  `--output_dim OUTPUT_DIM`:
                        output vector dimensionality <br/>
  `--hidden_size HIDDEN_SIZE`:
                        hidden state dimensionality <br/>
  `--no_bidirectional`:    bidirectional LSTM <br/>
  `--no_reverse_input`:    disable_input_reversal <br/>
  `--emb_dropout EMB_DROPOUT`:
                        input dropout rate <br/>
  `--rnn_dropout RNN_DROPOUT`:
                        dropout rate internal to encoder RNN <br/>
