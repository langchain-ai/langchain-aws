import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

from pydantic import SecretStr

if TYPE_CHECKING:
    from botocore.config import Config

from langchain_aws.utils import create_aws_client


def _format_triples(triples: List[dict]) -> List[str]:
    triple_template = "(:`{a}`)-[:`{e}`]->(:`{b}`)"
    triple_schema = []
    for t in triples:
        triple = triple_template.format(a=t["~from"], e=t["~type"], b=t["~to"])
        triple_schema.append(triple)

    return triple_schema


def _format_node_properties(n_labels: dict) -> List:
    node_properties = []

    for label, props_item in n_labels.items():
        props = props_item["properties"]
        np = {
            "properties": [
                {"property": k, "type": v["datatypes"][0]} for k, v in props.items()
            ],
            "labels": label,
        }
        node_properties.append(np)

    return node_properties


def _format_edge_properties(e_labels: dict) -> List:
    edge_properties = []

    for label, props_item in e_labels.items():
        props = props_item["properties"]
        np = {
            "type": label,
            "properties": [
                {"property": k, "type": v["datatypes"][0]} for k, v in props.items()
            ],
        }
        edge_properties.append(np)

    return edge_properties


class NeptuneQueryException(Exception):
    """Exception for the Neptune queries."""

    def __init__(self, exception: Union[str, Dict]):
        if isinstance(exception, dict):
            self.message = exception["message"] if "message" in exception else "unknown"
            self.details = exception["details"] if "details" in exception else "unknown"
        else:
            self.message = exception
            self.details = "unknown"

    def get_message(self) -> str:
        return self.message

    def get_details(self) -> Any:
        return self.details


class BaseNeptuneGraph(ABC):
    def __init__(
        self, property_descriptions: Optional[Dict[Tuple[str, str], str]] = None
    ):
        self.property_descriptions = property_descriptions or {}

    @property
    def get_schema(self) -> str:
        """Returns the schema of the Neptune database"""
        return self.schema

    @abstractmethod
    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    @abstractmethod
    def _get_summary(self) -> Dict:
        raise NotImplementedError()

    def _get_labels(self) -> Tuple[List[str], List[str]]:
        """Get node and edge labels from the Neptune statistics summary"""
        summary = self._get_summary()
        n_labels = summary["nodeLabels"]
        e_labels = summary["edgeLabels"]
        return n_labels, e_labels

    def _get_triples(self, e_labels: List[str]) -> List[str]:
        triple_query = """
        MATCH (a)-[e:`{e_label}`]->(b)
        WITH a,e,b LIMIT 3000
        RETURN DISTINCT labels(a) AS from, type(e) AS edge, labels(b) AS to
        LIMIT 10
        """

        triple_template = "(:`{a}`)-[:`{e}`]->(:`{b}`)"
        triple_schema = []
        for label in e_labels:
            q = triple_query.format(e_label=label)
            data = self.query(q)
            for d in data:
                triple = triple_template.format(
                    a=d["from"][0], e=d["edge"], b=d["to"][0]
                )
                triple_schema.append(triple)

        return triple_schema

    def _get_node_properties(
        self, n_labels: List[str], type_mapping: Dict[str, str]
    ) -> List:
        node_properties_query = """
        MATCH (a:`{n_label}`)
        RETURN properties(a) AS props
        LIMIT 100
        """
        node_properties = []
        for label in n_labels:
            q = node_properties_query.format(n_label=label)
            data = {"label": label, "properties": self.query(q)}
            s = set({})
            for p in data["properties"]:
                from typing import cast

                p_dict = cast(Dict[str, Any], p)
                props = cast(Dict[str, Any], p_dict["props"])
                for k, v in props.items():
                    s.add((k, type_mapping[type(v).__name__]))  # type: ignore

            np = {
                "properties": [{"property": k, "type": v} for k, v in s],
                "labels": label,
            }
            node_properties.append(np)

        return node_properties

    def _get_edge_properties(
        self, e_labels: List[str], type_mapping: Dict[str, str]
    ) -> List:
        edge_properties_query = """
        MATCH ()-[e:`{e_label}`]->()
        RETURN properties(e) AS props
        LIMIT 100
        """
        edge_properties = []
        for label in e_labels:
            q = edge_properties_query.format(e_label=label)
            data = {"label": label, "properties": self.query(q)}
            s = set({})
            for p in data["properties"]:
                from typing import cast

                p_dict = cast(Dict[str, Any], p)
                props = cast(Dict[str, Any], p_dict["props"])
                for k, v in props.items():
                    s.add((k, type_mapping[type(v).__name__]))  # type: ignore

            ep = {
                "type": label,
                "properties": [{"property": k, "type": v} for k, v in s],
            }
            edge_properties.append(ep)

        return edge_properties

    def _inject_property_descriptions(self, properties_list: List) -> List:
        """Inject property descriptions into node and edge properties."""
        if not self.property_descriptions:
            return properties_list

        for item in properties_list:
            for prop in item["properties"]:
                key_field = "labels" if "labels" in item else "type"
                key = (item[key_field], prop["property"])
                if key in self.property_descriptions:
                    prop["description"] = self.property_descriptions[key]
        return properties_list

    def _refresh_schema(self) -> None:
        """Refreshes the Neptune graph schema information."""

        types: Dict[str, str] = {
            "str": "STRING",
            "float": "DOUBLE",
            "int": "INTEGER",
            "list": "LIST",
            "dict": "MAP",
            "bool": "BOOLEAN",
        }
        n_labels, e_labels = self._get_labels()
        triple_schema = self._get_triples(e_labels)
        node_properties = self._inject_property_descriptions(
            self._get_node_properties(n_labels, types)
        )
        edge_properties = self._inject_property_descriptions(
            self._get_edge_properties(e_labels, types)
        )

        self.schema = f"""
        Node properties are the following:
        {node_properties}
        Relationship properties are the following:
        {edge_properties}
        The relationships are the following:
        {triple_schema}
        """


class NeptuneAnalyticsGraph(BaseNeptuneGraph):
    """Neptune Analytics wrapper for graph operations.

    Args:
        client: optional boto3 Neptune client
        credentials_profile_name: optional AWS profile name
        region_name: optional AWS region, e.g., us-west-2
        graph_identifier: the graph identifier for a Neptune Analytics graph

    Example:
        ```python
        graph = NeptuneAnalyticsGraph(
            graph_identifier='<my-graph-id>'
        )
        ```

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://docs.langchain.com/oss/python/security-policy for more information.

    """

    def __init__(
        self,
        graph_identifier: str,
        client: Any = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        aws_access_key_id: Optional[SecretStr] = None,
        aws_secret_access_key: Optional[SecretStr] = None,
        aws_session_token: Optional[SecretStr] = None,
        endpoint_url: Optional[str] = None,
        config: Optional["Config"] = None,
        property_descriptions: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> None:
        """Create a new Neptune Analytics graph wrapper instance."""
        super().__init__(property_descriptions)
        self.graph_identifier = graph_identifier

        if client is not None:
            self.client = client
        else:
            self.client = create_aws_client(
                region_name=region_name,
                credentials_profile_name=credentials_profile_name,
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                aws_session_token=aws_session_token,
                endpoint_url=endpoint_url,
                config=config,
                service_name="neptune-graph",
            )

        try:
            self._refresh_schema()
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": "Could not get schema for Neptune database",
                    "detail": str(e),
                }
            )

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query Neptune database."""
        try:
            resp = self.client.execute_query(
                graphIdentifier=self.graph_identifier,
                queryString=query,
                parameters=params,
                language="OPEN_CYPHER",
            )
            return json.loads(resp["payload"].read().decode("UTF-8"))["results"]
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": "An error occurred while executing the query.",
                    "details": str(e),
                }
            )

    def _get_summary(self) -> Dict:
        try:
            response = self.client.get_graph_summary(
                graphIdentifier=self.graph_identifier, mode="detailed"
            )
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": ("Summary API error occurred on Neptune Analytics"),
                    "details": str(e),
                }
            )

        try:
            summary = response["graphSummary"]
        except Exception:
            raise NeptuneQueryException(
                {
                    "message": "Summary API did not return a valid response.",
                    "details": response.content.decode(),
                }
            )
        else:
            return summary

    def _refresh_schema(self) -> None:
        """Refreshes the Neptune graph schema information."""
        pg_schema_query = """
        CALL neptune.graph.pg_schema()
        YIELD schema
        RETURN schema
        """

        data = self.query(pg_schema_query)
        raw_schema = data[0]["schema"]
        triple_schema = _format_triples(raw_schema["labelTriples"])
        node_properties = self._inject_property_descriptions(
            _format_node_properties(raw_schema["nodeLabelDetails"])
        )
        edge_properties = self._inject_property_descriptions(
            _format_edge_properties(raw_schema["edgeLabelDetails"])
        )

        self.schema = f"""
        Node properties are the following:
        {node_properties}
        Relationship properties are the following:
        {edge_properties}
        The relationships are the following:
        {triple_schema}
        """


class NeptuneGraph(BaseNeptuneGraph):
    """Neptune wrapper for graph operations.

    Args:
        host: endpoint for the database instance
        port: port number for the database instance, default is 8182
        use_https: whether to use secure connection, default is True
        client: optional boto3 Neptune client
        credentials_profile_name: optional AWS profile name
        region_name: optional AWS region, e.g., us-west-2
        sign: whether to sign the request payload, default is True
        aws_access_key_id: optional AWS access key ID
        aws_secret_access_key: optional AWS secret access key
        aws_session_token: optional AWS session token
        endpoint_url: optional custom endpoint URL
        config: optional botocore Config object

    Example:
        ```python
        graph = NeptuneGraph(
            host='<my-cluster>',
            port=8182
        )
        ```

    *Security note*: Make sure that the database connection uses credentials
        that are narrowly-scoped to only include necessary permissions.
        Failure to do so may result in data corruption or loss, since the calling
        code may attempt commands that would result in deletion, mutation
        of data if appropriately prompted or reading sensitive data if such
        data is present in the database.
        The best way to guard against such negative outcomes is to (as appropriate)
        limit the permissions granted to the credentials used with this tool.

        See https://docs.langchain.com/oss/python/security-policy for more information.

    """

    def __init__(
        self,
        host: str,
        port: int = 8182,
        use_https: bool = True,
        client: Any = None,
        credentials_profile_name: Optional[str] = None,
        region_name: Optional[str] = None,
        sign: bool = True,
        aws_access_key_id: Optional[SecretStr] = None,
        aws_secret_access_key: Optional[SecretStr] = None,
        aws_session_token: Optional[SecretStr] = None,
        endpoint_url: Optional[str] = None,
        config: Optional["Config"] = None,
        property_descriptions: Optional[Dict[Tuple[str, str], str]] = None,
    ) -> None:
        """Create a new Neptune graph wrapper instance."""
        super().__init__(property_descriptions)

        try:
            if client is not None:
                self.client = client
            else:
                import boto3

                any_creds = bool(
                    credentials_profile_name
                    or aws_access_key_id
                    or aws_secret_access_key
                    or aws_session_token
                )

                if not any_creds:
                    session = boto3.Session()
                elif credentials_profile_name:
                    session = boto3.Session(profile_name=credentials_profile_name)
                elif aws_access_key_id and aws_secret_access_key:
                    session_params = {
                        "aws_access_key_id": aws_access_key_id.get_secret_value(),
                        "aws_secret_access_key": (
                            aws_secret_access_key.get_secret_value()
                        ),
                    }
                    if aws_session_token:
                        session_params["aws_session_token"] = (
                            aws_session_token.get_secret_value()
                        )
                    # session_params contains valid boto3.Session parameters but type
                    # stubs are overly restrictive
                    session = boto3.Session(**session_params)  # type: ignore[arg-type]
                else:
                    raise ValueError(
                        "If providing credentials, both aws_access_key_id and "
                        "aws_secret_access_key must be specified."
                    )

                client_params = {}
                if region_name:
                    client_params["region_name"] = region_name

                if endpoint_url is not None:
                    client_params["endpoint_url"] = endpoint_url
                else:
                    protocol = "https" if use_https else "http"
                    client_params["endpoint_url"] = f"{protocol}://{host}:{port}"

                if config is not None:
                    # client_params dict expects string keys but Config is correct
                    # type here
                    client_params["config"] = config  # type: ignore[assignment]

                if not sign:
                    from botocore import UNSIGNED
                    from botocore.config import Config

                    if "config" in client_params:
                        from typing import cast

                        existing_config = cast("Config", client_params["config"])
                        # Config.merge() returns Config but type system expects string
                        client_params["config"] = existing_config.merge(
                            Config(signature_version=UNSIGNED)
                        )  # type: ignore[assignment]
                    else:
                        # Config object is correct type here but type system expects
                        # string
                        client_params["config"] = Config(signature_version=UNSIGNED)  # type: ignore[assignment]

                # boto3 type stubs don't recognize neptunedata service
                self.client = session.client("neptunedata", **client_params)  # type: ignore[call-overload]

        except ImportError:
            raise ModuleNotFoundError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        except Exception as e:
            if type(e).__name__ == "UnknownServiceError":
                raise ModuleNotFoundError(
                    "NeptuneGraph requires a boto3 version 1.28.38 or greater."
                    "Please install it with `pip install -U boto3`."
                ) from e
            else:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    "profile name are valid."
                ) from e

        try:
            self._refresh_schema()
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": "Could not get schema for Neptune database",
                    "detail": str(e),
                }
            )

    def query(self, query: str, params: dict = {}) -> List[Dict[str, Any]]:
        """Query Neptune database."""
        try:
            if params:
                return self.client.execute_open_cypher_query(
                    openCypherQuery=query, parameters=json.dumps(params)
                )["results"]
            else:
                return self.client.execute_open_cypher_query(openCypherQuery=query)[
                    "results"
                ]
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": "An error occurred while executing the query.",
                    "details": str(e),
                }
            )

    def _get_summary(self) -> Dict:
        try:
            response = self.client.get_propertygraph_summary()
        except Exception as e:
            raise NeptuneQueryException(
                {
                    "message": (
                        "Summary API is not available for this instance of Neptune,"
                        "ensure the engine version is >=1.2.1.0"
                    ),
                    "details": str(e),
                }
            )

        try:
            summary = response["payload"]["graphSummary"]
        except Exception:
            raise NeptuneQueryException(
                {
                    "message": "Summary API did not return a valid response.",
                    "details": response.content.decode(),
                }
            )
        else:
            return summary
