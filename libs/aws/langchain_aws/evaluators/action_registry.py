import typing
from abc import ABC, abstractmethod
from collections import UserDict

from langchain_aws.evaluators.inference_evaluator_exception import (
    InferenceEvaluatorError,
)


class ActionRegistry(UserDict[str, "ActionInterface"]):
    """
    Action registry.
    """

    def __setitem__(self, key: str, value: "ActionInterface"):  # type: ignore
        """
        Set actions

        :param key: action name
        :param value: action class
        """
        if "ActionInterface" in (c.__name__ for c in value.__mro__):  # type: ignore
            super().__setitem__(key, value)
        else:
            raise ValueError(
                "Please inherit the action from EvaluationAlgorithmInterface"
            )

    def get_action(self, action_name: str) -> "ActionInterface":
        """
        Get action class with name

        :param action_name: action name
        :return: action class
        """
        if action_name in self:
            return self[action_name]
        else:
            raise KeyError(f"Unknown action {action_name}")


ACTION_REGISTRY = ActionRegistry()


class ActionInterface(ABC):
    """
    Interface class for action.

    All the action classes inheriting this interface will be registered in the registry.
    """

    def __init__(self) -> None:
        super().__init__()
        try:
            import boto3

            self.events_client = boto3.client("events")

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

    def __init_subclass__(cls, **kwargs):  # type: ignore
        """
        Method to register algorithms
        """
        super().__init_subclass__(**kwargs)
        ACTION_REGISTRY[cls.__name__] = cls

    @abstractmethod
    def run(self, input_text: str, additional_parameters: dict) -> typing.Any:
        """Run the custom defined actions"""
        pass


class Deny(ActionInterface):
    def run(self, input_text: str, configuration: dict) -> None:
        raise InferenceEvaluatorError(configuration["alternative_response"])


class Instruct(ActionInterface):
    def _event_bus_exists(self, name: str) -> bool:
        try:
            self.events_client.describe_event_bus(Name=name)
            return True
        except Exception:
            return False

    def _create_event_bus(self, payload: dict) -> None:
        if "Name" not in payload:
            raise ValueError(
                "Name for the event bus is not provided in "
                "the inference evaluator config"
            )

        name = payload["Name"]
        if not self._event_bus_exists(name):
            self.events_client.create_event_bus(**payload)

    def _put_events(self, payload: dict) -> None:
        self.events_client.put_events(**payload)

    def run(self, input_text: str, configuration: dict) -> str:
        api_invokes = configuration["additional_parameters"]["api_invokes"]

        if "create_event_bus" in api_invokes:
            create_event_bus_payload = api_invokes["create_event_bus"]
            self._create_event_bus(create_event_bus_payload)

        if "put_events" in api_invokes:
            put_events_payload = api_invokes["put_events"]
            self._put_events(put_events_payload)

        return input_text


class Redact(ActionInterface):
    def run(self, input_text: str, configuration: dict) -> str:
        message = configuration["additional_parameters"]["message"]
        return "{0}\n{1}".format(input_text, message)
