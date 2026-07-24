"use client";

import { useState, useRef, useEffect } from "react";
import { Input } from "./ui/Input";

export interface InlineEditProps {
  value: string;
  onSave: (val: string) => Promise<void>;
  className?: string;
  inputClassName?: string;
}

export function InlineEdit({ value, onSave, className, inputClassName }: InlineEditProps) {
  const [isEditing, setIsEditing] = useState(false);
  const [currentValue, setCurrentValue] = useState(value);
  const [isSaving, setIsSaving] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  // Sync value when parent passes new value (optimistic update from parent or real fetch)
  useEffect(() => {
    if (!isEditing) {
      setCurrentValue(value);
    }
  }, [value, isEditing]);

  useEffect(() => {
    if (isEditing && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isEditing]);

  const handleSave = async () => {
    if (!currentValue.trim() || currentValue === value) {
      setIsEditing(false);
      setCurrentValue(value);
      return;
    }
    if (currentValue.length > 100) {
      return;
    }
    setIsSaving(true);
    try {
      await onSave(currentValue);
      setIsEditing(false);
    } catch (err) {
      setCurrentValue(value);
      setIsEditing(false);
      // parent handles toast on failure
    } finally {
      setIsSaving(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") handleSave();
    if (e.key === "Escape") {
      setIsEditing(false);
      setCurrentValue(value);
    }
  };

  const handleTriggerKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      e.stopPropagation();
      setIsEditing(true);
    }
  };

  if (!isEditing) {
    return (
      <span
        role="button"
        tabIndex={0}
        aria-label="Edit experiment name"
        onClick={(e) => {
          e.stopPropagation();
          setIsEditing(true);
        }}
        onKeyDown={handleTriggerKeyDown}
        className={`cursor-pointer focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-neural/50 rounded hover:underline ${className || ""}`}
        title="Click to edit"
      >
        {value}
      </span>
    );
  }

  return (
    <Input
      ref={inputRef}
      value={currentValue}
      onChange={(e) => setCurrentValue(e.target.value)}
      onBlur={handleSave}
      onKeyDown={handleKeyDown}
      disabled={isSaving}
      maxLength={100}
      aria-label="Edit experiment name"
      className={`h-6 text-sm px-1 py-0 w-full ${inputClassName || ""}`}
      onClick={(e) => e.stopPropagation()}
    />
  );
}