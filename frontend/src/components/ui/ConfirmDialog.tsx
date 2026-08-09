"use client";

import { Button } from "@/components/ui/Button";
import { Modal } from "@/components/ui/Modal";
import { type ReactNode } from "react";

export interface ConfirmDialogProps {
  open: boolean;
  onClose: () => void;
  title?: string;
  subtitle?: string;
  children?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: "danger" | "primary" | "secondary";
  onConfirm: () => void;
  isLoading?: boolean;
}

export function ConfirmDialog({
  confirmLabel = "Confirm",
  cancelLabel = "Cancel",
  variant = "danger",
  onConfirm,
  isLoading = false,
  children,
  ...modalProps
}: ConfirmDialogProps) {
  const handleConfirm = () => {
    onConfirm();
  };

  const footer = (
    <>
      <Button variant="ghost" size="sm" onClick={modalProps.onClose} disabled={isLoading}>
        {cancelLabel}
      </Button>
      <Button
        variant={variant === "danger" ? "danger" : variant === "primary" ? "primary" : "secondary"}
        size="sm"
        onClick={handleConfirm}
        disabled={isLoading}
      >
        {isLoading ? "Processing..." : confirmLabel}
      </Button>
    </>
  );

  return <Modal {...modalProps} footer={footer}>{children}</Modal>;
}
