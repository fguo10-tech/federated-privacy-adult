Tiny SYNTHETIC test fixtures for the federated track. No real person's data is here.

* `adult/adult.data`, `adult/adult.test` — random rows in the exact UCI Adult file format
  (leading spaces, `?` for missing, a `|1x3 Cross validator` comment line and `>50K.` labels
  with a trailing period in the test file). Values were drawn from the published schema
  vocabulary with a fixed NumPy seed; they are not census records.
* `leaf/femnist_fixture_0.json` — three fake writers with random 0/1 "images" and random
  labels in LEAF JSON format, for the FEMNIST reader test.
