import pandas
import modelop.schema.infer as infer
import modelop.utils as utils

logger = utils.configure_logger()

MONITORING_PARAMETERS = {}

# modelop.init
def init(job_json):
    """A function to extract input schema from job JSON.

    Args:
        job_json (str): job JSON in a string format.
    """

    # Extract input schema from job JSON
    input_schema_definition = infer.extract_input_schema(job_json)

    logger.info("Input schema definition: %s", input_schema_definition)

    # Get monitoring parameters from schema
    global MONITORING_PARAMETERS
    MONITORING_PARAMETERS = infer.set_monitoring_parameters(
        schema_json=input_schema_definition, check_schema=True
    )

    logger.info("numerical_columns: %s", MONITORING_PARAMETERS["numerical_columns"])


# modelop.metrics
def metrics(dataframe):

    # Compute outlier metrics for each numerical column by IQR method
    values = [
        detect_outliers_IQR(dataframe=dataframe, column=field) 
        for field in MONITORING_PARAMETERS["numerical_columns"]
    ]

    # Top-level metrics
    result = {
        "{}_number_outliers".format(val["field"]): val["number_outliers"] for val in values
    }

    # Add test results
    result["outliers"] = [
        {
            "test_name": "Outliers IQR",
            "test_category": "outliers",
            "test_type": "IQR",
            "test_id": "outliers_IQR",
            "values": values,
        }
    ]
    yield result


def detect_outliers_IQR(dataframe: pandas.DataFrame, column: str)->dict:
    """A function to compute IQR metrics given a dataframe and a numerical column name.

    Args:
        dataframe (pandas.DataFrame): Input Dataframe
        column (str): Numerical column

    Returns:
        dict: Summary metrics.
    """

    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)
    IQR = Q3-Q1

    outliers_condition=((dataframe[column]<(Q1-1.5*IQR)) | (dataframe[column]>(Q3+1.5*IQR)))

    outliers_list = dataframe[outliers_condition][column].values

    outliers = {
        "field": column,
        "Q1": Q1, 
        "Q3": Q3, 
        "IQR": IQR,
        "IQR_lower_bound": Q1-1.5*IQR,
        "IQR_upper_bound": Q3+1.5*IQR,
        "IQInterval":"[{},{}]".format((Q1-1.5*IQR),(Q3+1.5*IQR)),
        "number_outliers": len(outliers_list),
        "max_outlier": max(outliers_list) if (len(outliers_list)>0) else None,
        "min_outlier": min(outliers_list) if (len(outliers_list)>0) else None,
    }

    return outliers