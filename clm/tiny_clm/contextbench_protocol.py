"""Optional diagnostic adapters. Original agents and their defaults are untouched."""
from clm_contextbench import ClmStream
from contextbench.harbor.agents.base import ContextBenchAgent

from .answer_format import FormatReminderMixin, ToolFormatReminderMixin


class FormatReminderClm(FormatReminderMixin, ClmStream):
    @staticmethod
    def name():
        return "clm-contextbench-format-v1"


class FormatReminderAppendOnly(FormatReminderMixin, ContextBenchAgent):
    @staticmethod
    def name():
        return "contextbench-bash-format-v1"


class ToolFormatReminderClm(ToolFormatReminderMixin, ClmStream):
    @staticmethod
    def name():
        return "clm-contextbench-tool-format-v1"


class ToolFormatReminderAppendOnly(ToolFormatReminderMixin, ContextBenchAgent):
    @staticmethod
    def name():
        return "contextbench-bash-tool-format-v1"
