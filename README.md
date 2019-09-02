# rustboard

Rough implementation of the TensorBoard HTTP APIs in Rust, for fun.

On a ~300 MB logdir, standard TensorBoard 1.14 takes about 750 seconds
to read the contents into memory; `rustboard` takes just 1.3 seconds.

This is not an official Google product.
