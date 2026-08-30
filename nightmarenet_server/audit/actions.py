"""SOC 2 CC7.2 audit action vocabulary."""

from enum import Enum


class AuditAction(str, Enum):
    """Immutable audit action types."""

    CREATE = "CREATE"
    READ = "READ"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    LOGIN = "LOGIN"
    LOGOUT = "LOGOUT"
    EXPORT = "EXPORT"
    PERMISSION_CHANGE = "PERMISSION_CHANGE"


MUTATION_METHOD_ACTIONS = {
    "POST": AuditAction.CREATE,
    "PUT": AuditAction.UPDATE,
    "PATCH": AuditAction.UPDATE,
    "DELETE": AuditAction.DELETE,
}
