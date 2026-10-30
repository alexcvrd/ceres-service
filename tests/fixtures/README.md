# Ceres test fixtures

A frozen slice of one recorded DDS route run: the store rows (`items.jsonl`, `items.npy`), the questions and their query
vectors, the golden candidate pools and answers (`golden/`), the profiles of that route (`profiles/`) and the schemas the
packets were validated against (`schemas/`). `MANIFEST.json` records how the slice was built and the sha256 and size of every
file here; `tests/test_fixtures.py` checks them.

The profiles under `profiles/` are the exact bytes of the profiles that produced the
recorded answers. Their sha256 values make up the route id the golden answers were judged under, so they are kept byte for
byte, including their original comments and the provenance paths they name (`inherited_runner`, `platform_config`,
`answer_contract`). Ceres does not read those paths. Paths of the machine that built the slice are written as `<dds>/…`.
