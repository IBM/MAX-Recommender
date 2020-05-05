## How to prepare your data for training

Follow the instructions in this document to prepare your data for model training.
- [Prerequisites](#prerequisites)
- [Preparing your data](#preparing-your-data)
- [Organize data directory](#organize-data-directory)

## Preparing your data

The model trains on CSV files with the following format: `User ID, Item ID, Rating, Timestamp`

## Organize data directory

A trainable model should adhere to the standard directory structure below:

```
|-- data_directory
    |-- assets
    |-- data
    |-- initial_model
```

1. `assets` holds ancillary files required for training (typically these are generated during the data preparation phase).
2. `data` folder holds the data required for training.
3. `initial_model` folder holds the initial checkpoint files to initiate training.

If a particular directory is not required, it can be omitted.
