# This code snippet will load the referenced package and return a DataFrame.
# If the code is run in a PySpark environment, then the code will return a
# Spark DataFrame. If not, the code will return a Pandas DataFrame. You can
# copy this code snippet to another code file as needed.
from azureml.dataprep.package import run


# Use this DataFrame for further processing
df = run('sampleReviews.dprep', dataflow_idx=0)

rows, columns = df.shape

for i in range(0, rows):
    try:
        print(df.iloc[i,0] + ' ' + df.iloc[i,1])
    except UnicodeEncodeError:
        pass