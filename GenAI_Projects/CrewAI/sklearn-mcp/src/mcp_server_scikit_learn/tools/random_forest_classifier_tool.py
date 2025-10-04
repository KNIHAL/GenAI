import json
from typing import Any, Optional

import mcp.types as types
import polars as pl
from pydantic import BaseModel, ConfigDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from ..configure_logger import make_logger
from ..utils.string_util import string_to_list_dict
from .tools import MCPServerScikitLearnTools

logger = make_logger(__name__)


class RandomForestClassifierInputSchema(BaseModel):
    model_config = ConfigDict(
        validate_assignment=True,
        frozen=True,
        extra="forbid",
        arbitrary_types_allowed=True,
    )

    training_dataset: list[dict[str, str | int | float | bool | None]]
    target_column: str
    prediction_dataset: list[dict[str, str | int | float | bool | None]]
    column_to_ignore: Optional[list[str]] = None
    predict_proba: bool = False
    # Random Forest hyperparameters
    n_estimators: Optional[int] = 100
    criterion: Optional[str] = "gini"
    max_depth: Optional[int | None] = None
    min_samples_split: Optional[int | float] = 2
    min_samples_leaf: Optional[int | float] = 0
    min_weight_fraction_leaf: Optional[float] = 0.0
    max_features: Optional[str | int | float] = "sqrt"
    max_leaf_nodes: Optional[int | None] = None
    min_impurity_decrease: Optional[float] = 0.0
    bootstrap: Optional[bool] = True
    oob_score: Optional[bool] = False
    class_weight: Optional[str | dict | list[dict] | None] = None
    ccp_alpha: Optional[float] = 0.0

    @staticmethod
    def input_schema() -> dict:
        return {
            "type": "object",
            "properties": {
                "training_dataset": {
                    "type": "string",
                    "description": "The dataset to train in Base64 encoded string or dumped JSON.",
                },
                "target_column": {
                    "type": "string",
                    "description": "The target column to predict from training_dataset.",
                },
                "prediction_dataset": {
                    "type": "string",
                    "description": "The dataset to predict in Base64 encoded string or dumped JSON.",
                },
                "column_to_ignore": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "The columns to ignore in training and prediction.",
                },
                "predict_proba": {
                    "type": "bool",
                    "description": "Whether to predict probabilities or not.",
                },
                "n_estimators": {
                    "type": "integer",
                    "description": "The number of trees in the forest.",
                },
                "criterion": {
                    "type": "string",
                    "enum": ["gini", "entropy", "log_loss"],
                    "description": "The function to measure the quality of a split.",
                },
                "max_depth": {
                    "type": ["integer", "null"],
                    "description": "The maximum depth of the tree.",
                },
                "min_samples_split": {
                    "type": ["integer", "number"],
                    "description": "The minimum number of samples required to split an internal node.",
                },
                "min_samples_leaf": {
                    "type": ["integer", "number"],
                    "description": "The minimum number of samples required to be at a leaf node.",
                },
                "min_weight_fraction_leaf": {
                    "type": "number",
                    "description": "The minimum weighted fraction of the sum total of weights required to be at a leaf node.",
                },
                "max_features": {
                    "type": ["string", "integer", "number"],
                    "description": "The number of features to consider when looking for the best split.",
                },
                "max_leaf_nodes": {
                    "type": ["integer", "null"],
                    "description": "Grow trees with max_leaf_nodes in best-first fashion.",
                },
                "min_impurity_decrease": {
                    "type": "number",
                    "description": "A node will be split if this split induces a decrease of the impurity greater than or equal to this value.",
                },
                "bootstrap": {
                    "type": "boolean",
                    "description": "Whether bootstrap samples are used when building trees.",
                },
                "oob_score": {
                    "type": "boolean",
                    "description": "Whether to use out-of-bag samples to estimate the generalization score.",
                },
                "class_weight": {
                    "type": ["string", "object", "array"],
                    "description": "Weights associated with classes in the form {class_label: weight}.",
                },
                "ccp_alpha": {
                    "type": "number",
                    "description": "Complexity parameter used for Minimal Cost-Complexity Pruning.",
                },
            },
        }

    @staticmethod
    def from_str(
        training_dataset: str,
        target_column: str,
        prediction_dataset: str,
        column_to_ignore: Optional[list[str]] = None,
        predict_proba: bool = False,
        n_estimators: Optional[int] = 100,
        criterion: Optional[str] = "gini",
        max_depth: Optional[int | None] = None,
        min_samples_split: Optional[int | float] = 2,
        min_samples_leaf: Optional[int | float] = 0,
        min_weight_fraction_leaf: Optional[float] = 0.0,
        max_features: Optional[str | int | float] = "sqrt",
        max_leaf_nodes: Optional[int | None] = None,
        min_impurity_decrease: Optional[float] = 0.0,
        bootstrap: Optional[bool] = True,
        oob_score: Optional[bool] = False,
        class_weight: Optional[str | dict | list[dict] | None] = None,
        ccp_alpha: Optional[float] = 0.0,
    ) -> "RandomForestClassifierInputSchema":
        structured_training_dataset = string_to_list_dict(training_dataset)
        structured_prediction_dataset = string_to_list_dict(prediction_dataset)

        return RandomForestClassifierInputSchema(
            training_dataset=structured_training_dataset,
            target_column=target_column,
            prediction_dataset=structured_prediction_dataset,
            column_to_ignore=column_to_ignore,
            predict_proba=predict_proba,
            n_estimators=n_estimators,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            min_weight_fraction_leaf=min_weight_fraction_leaf,
            max_features=max_features,
            max_leaf_nodes=max_leaf_nodes,
            min_impurity_decrease=min_impurity_decrease,
            bootstrap=bootstrap,
            oob_score=oob_score,
            class_weight=class_weight,
            ccp_alpha=ccp_alpha,
        )


random_forest_classifier_tool = types.Tool(
    name=MCPServerScikitLearnTools.RANDOM_FOREST_CLASSIFIER.value,
    description="Train and predict with Random Forest Classifier.",
    inputSchema=RandomForestClassifierInputSchema.input_schema(),
)


async def handle_random_forest_classifier_tool(
    arguments: dict[str, Any],
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    logger.info("Handling random forest classifier tool")

    training_dataset = arguments["training_dataset"]
    target_column = arguments["target_column"]
    prediction_dataset = arguments["prediction_dataset"]
    column_to_ignore = arguments.get("column_to_ignore", None)
    predict_proba = arguments.get("predict_proba", False)

    # Get hyperparameters from arguments
    n_estimators = arguments.get("n_estimators", 100)
    criterion = arguments.get("criterion", "gini")
    max_depth = arguments.get("max_depth", None)
    min_samples_split = arguments.get("min_samples_split", 2)
    min_samples_leaf = arguments.get("min_samples_leaf", 1)
    min_weight_fraction_leaf = arguments.get("min_weight_fraction_leaf", 0.0)
    max_features = arguments.get("max_features", "sqrt")
    max_leaf_nodes = arguments.get("max_leaf_nodes", None)
    min_impurity_decrease = arguments.get("min_impurity_decrease", 0.0)
    bootstrap = arguments.get("bootstrap", True)
    oob_score = arguments.get("oob_score", False)
    class_weight = arguments.get("class_weight", None)
    ccp_alpha = arguments.get("ccp_alpha", 0.0)

    random_forest_classifier_input = RandomForestClassifierInputSchema.from_str(
        training_dataset,
        target_column,
        prediction_dataset,
        column_to_ignore,
        predict_proba,
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        min_impurity_decrease=min_impurity_decrease,
        bootstrap=bootstrap,
        oob_score=oob_score,
        class_weight=class_weight,
        ccp_alpha=ccp_alpha,
    )

    train_df = pl.DataFrame(random_forest_classifier_input.training_dataset)
    prediction_df = pl.DataFrame(random_forest_classifier_input.prediction_dataset)
    training_columns = []
    numerical_columns_scalars = {}
    categorical_columns_encoders = {}

    for col in train_df.columns:
        if col == random_forest_classifier_input.target_column:
            continue
        if random_forest_classifier_input.column_to_ignore and col in random_forest_classifier_input.column_to_ignore:
            continue

        training_columns.append(col)
        if train_df[col].dtype in [pl.Int64, pl.Float64]:
            scaler = MinMaxScaler()
            train_df = train_df.with_columns(
                [pl.Series(col, scaler.fit_transform(train_df[col].to_numpy().reshape(-1, 1)).flatten())]
            )
            numerical_columns_scalars[col] = scaler
        if train_df[col].dtype == pl.Utf8:
            label_encoder = LabelEncoder()
            train_df = train_df.with_columns(pl.Series(col, label_encoder.fit_transform(train_df[col].to_list())))
            categorical_columns_encoders[col] = label_encoder

    for col in prediction_df.columns:
        if col == random_forest_classifier_input.target_column:
            continue
        if random_forest_classifier_input.column_to_ignore and col in random_forest_classifier_input.column_to_ignore:
            continue
        if col not in training_columns:
            continue
        if col in numerical_columns_scalars.keys():
            prediction_df = prediction_df.with_columns(
                [
                    pl.Series(
                        col,
                        numerical_columns_scalars[col]
                        .transform(prediction_df[col].to_numpy().reshape(-1, 1))
                        .flatten(),
                    )
                ]
            )
        if col in categorical_columns_encoders.keys():
            prediction_df = prediction_df.with_columns(
                pl.Series(col, categorical_columns_encoders[col].transform(prediction_df[col].to_list()))
            )

    x = train_df.drop(random_forest_classifier_input.target_column)
    if random_forest_classifier_input.column_to_ignore:
        x = x.drop(*random_forest_classifier_input.column_to_ignore)
    x = x.to_numpy()
    y = train_df[random_forest_classifier_input.target_column].to_numpy()

    estimator = RandomForestClassifier(
        n_estimators=random_forest_classifier_input.n_estimators,
        criterion=random_forest_classifier_input.criterion,
        max_depth=random_forest_classifier_input.max_depth,
        min_samples_split=random_forest_classifier_input.min_samples_split,
        min_samples_leaf=random_forest_classifier_input.min_samples_leaf,
        min_weight_fraction_leaf=random_forest_classifier_input.min_weight_fraction_leaf,
        max_features=random_forest_classifier_input.max_features,
        max_leaf_nodes=random_forest_classifier_input.max_leaf_nodes,
        min_impurity_decrease=random_forest_classifier_input.min_impurity_decrease,
        bootstrap=random_forest_classifier_input.bootstrap,
        oob_score=random_forest_classifier_input.oob_score,
        n_jobs=-1,
        random_state=42,
        class_weight=random_forest_classifier_input.class_weight,
        ccp_alpha=random_forest_classifier_input.ccp_alpha,
    )
    estimator.fit(x, y)

    prediction_x = (
        prediction_df.drop(random_forest_classifier_input.target_column)
        if random_forest_classifier_input.target_column in prediction_df.columns
        else prediction_df
    )
    if random_forest_classifier_input.column_to_ignore:
        prediction_x = prediction_x.drop(*random_forest_classifier_input.column_to_ignore)
    prediction_x = prediction_x.to_numpy()

    if random_forest_classifier_input.predict_proba:
        proba = estimator.predict_proba(prediction_x)
        classes = estimator.classes_.tolist()
        result = [{classes[i]: proba[j][i] for i in range(len(classes))} for j in range(len(proba))]
    else:
        result = estimator.predict(prediction_x).tolist()

    return [
        types.TextContent(
            type="text",
            text=json.dumps(
                {
                    "description": "Result of Random Forest Classifier.",
                    "predictions": result,
                }
            ),
        )
    ]
