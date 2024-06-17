import dask.dataframe as dd
import pandas as pd
import dask.array as da

# Create a Dask DataFrame
df = dd.from_pandas(pd.DataFrame({"x": [1, 2, 3], "y": [4, 5, 6]}), npartitions=2)


# Define the function to be applied on each row
def process_row(row):
    x = row["x"]
    y = row["y"]
    z = da.sqrt(x + y)  # Assuming some computation on x and y
    label = f"label_{x}_{y}"  # Assuming some label generation based on x and y
    return pd.Series(
        {"z": z.flatten(), "label": label}
    )  # Compute and flatten the Dask array


# Apply the function to each row of the Dask DataFrame
result = df.apply(process_row, axis=1, meta=pd.Series({"z": [], "label": str}))

# Compute the result
result = result.compute()

print(result)
