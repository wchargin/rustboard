# rustboard

Rough implementation of the TensorBoard HTTP APIs in Rust, for fun.

On a ~300 MB logdir of pure scalar data, standard TensorBoard 1.14 takes
about 50 seconds to read the contents into memory; `rustboard` takes
just 1.3 seconds.

On a [~1.3 GB logdir of real-world data][ds-user] and a total of
23 806 644 data points across 8 runs and 11 event files, TensorBoard
1.14 takes about 1117 seconds to read the contents into memory;
`rustboard` takes just 2.3 seconds.

This is not an official Google product.

[ds-user]: https://github.com/tensorflow/tensorboard/issues/766#issuecomment-524637583
