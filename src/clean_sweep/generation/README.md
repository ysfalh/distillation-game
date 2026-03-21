Generation module contract:

- explicit fixed batch size
- no auto-scaling
- no hidden backend toggles
- standard decoding
- antidistillation teacher decoding
- product-of-experts decoding
- optional strategic teacher decoding if we keep parity with the current sweep surface

Implementation note:
teacher generation should emit compact trace records keyed by `example_id`, with prompt text stored once per split when possible.
