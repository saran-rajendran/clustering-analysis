from typing import ClassVar, Dict
import pandas as pd
from pydantic import BaseModel, Field
from neontology import BaseNode, BaseRelationship, init_neontology, auto_constrain

import config

init_neontology(
    neo4j_uri=config.NEO4J_URI,
    neo4j_username=config.NEO4J_USERNAME,
    neo4j_password=config.NEO4J_PASSWORD
)


class PinNode(BaseNode):
    __primarylabel__: ClassVar[str] = "ID"
    __primaryproperty__: ClassVar[str] = "pin_id"

    pin_id: str
    pin_type: str
    kostenstelle: int
    arbeitsfolge: str
    bearbeitungsschritt: str
    datum: str
    zeit: str
    ergebnis: str
    ausfuerungszeit: float
    anzeigezeit: int


class FeatureNode(BaseNode):
    __primarylabel__: ClassVar[str] = "Feature"
    __primaryproperty__: ClassVar[str] = "name"
    # pin_id: str
    name: str
    value: float
    upper: float
    lower: float


class LabelNode(BaseNode):
    __primarylabel__: ClassVar[str] = "Label"
    __primaryproperty__: ClassVar[str] = "label"
    label: int


class HasFeatureRelationship(BaseRelationship):
    __relationshiptype__: ClassVar[str] = "HAS_FEATURE"

    source: PinNode
    target: FeatureNode


class HasLabelRelationship(BaseRelationship):
    __relationshiptype__: ClassVar[str] = "HAS_LABEL"

    source: PinNode
    target: LabelNode


def upload_to_neo4j(df):
    labels_set = set()
    for _, row in df.iterrows():
        pin_node = PinNode(pin_id=row['Pin_ID'], pin_type=row['Pin_Type'], kostenstelle=row['Kostenstelle'], arbeitsfolge=row['Arbeitsfolge'],
                           bearbeitungsschritt=row['Bearbeitungsschritt'], datum=row[
                               'Datum'], zeit=row['Zeit'], ergebnis=row['Ergebnis'],
                           ausfuerungszeit=row['Ausfuehrungszeit'], anzeigezeit=row['Anzeigezeit'])
        pin_node.create()

        feature_nodes = []
        for feature in config.features_with_threshold.keys():
            feature_node = FeatureNode(
                name=feature,
                value=row[config.features_with_threshold[feature][0]],
                upper=row[config.features_with_threshold[feature][1]],
                lower=row[config.features_with_threshold[feature][2]]
            )
            feature_node.create()
            feature_nodes.append(feature_node)

        if row['Label'] not in labels_set:
            label_node = LabelNode(label=row['Label'])
            label_node.create()
            labels_set.add(row['Label'])

        for feature_node in feature_nodes:
            rel = HasFeatureRelationship(source=pin_node, target=feature_node)
            rel.merge()

        rel = HasLabelRelationship(source=pin_node, target=label_node)
        rel.merge()
