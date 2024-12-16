# Tests


## Setup

The tests are written for pytest. They use an example dataset, consisting of 5 zip
files:
- `anchor.zip`: max and median projection for 1 tile of the anchor round
- `fluorescence.zip`: max and median projection for 1 tile of mCherry and DAPI overview
- `hybridisation.zip`: max and median projection for 1 tile of the hybridisation round
- `barcode_all_rounds.zip`: max and median projection for 1 tile of 10 barcode rounds
- `genes_all_rounds.zip`: max and median projection for 1 tile of 7 gene rounds


Once these files are downloaded (and put in the same folder), the path to the folder
must be set in `iss-preprocess/tests/pytest_fixture.py` in the `RAW_DATA_DIR` variable.

## Running the tests

To run the tests, navigate to the `iss-preprocess` folder and run:

```bash
pytest tests
```
