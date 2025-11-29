import React from 'react';
import { FiCheck, FiAlertCircle, FiX } from 'react-icons/fi';

interface ValidationFeedbackProps {
  error?: string | string[];
  touched?: boolean;
  success?: boolean;
  warning?: string;
  className?: string;
}

/**
 * ValidationFeedback Component
 *
 * Displays validation feedback below form inputs
 */
export const ValidationFeedback: React.FC<ValidationFeedbackProps> = ({
  error,
  touched,
  success,
  warning,
  className = '',
}) => {
  // Don't show anything if not touched and no success/warning
  if (!touched && !success && !warning) {
    return null;
  }

  // Error state
  if (error && touched) {
    const errors = Array.isArray(error) ? error : [error];
    return (
      <div className={`mt-1 ${className}`}>
        {errors.map((err, index) => (
          <div key={index} className="flex items-start gap-1.5 text-sm text-danger-600 dark:text-danger-400">
            <FiAlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
            <span>{err}</span>
          </div>
        ))}
      </div>
    );
  }

  // Warning state
  if (warning) {
    return (
      <div className={`mt-1 flex items-start gap-1.5 text-sm text-warning-600 dark:text-warning-400 ${className}`}>
        <FiAlertCircle className="w-4 h-4 mt-0.5 flex-shrink-0" />
        <span>{warning}</span>
      </div>
    );
  }

  // Success state
  if (success && touched) {
    return (
      <div className={`mt-1 flex items-center gap-1.5 text-sm text-success-600 dark:text-success-400 ${className}`}>
        <FiCheck className="w-4 h-4 flex-shrink-0" />
        <span>Looks good!</span>
      </div>
    );
  }

  return null;
};

/**
 * Input with Validation
 */
interface ValidatedInputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string | string[];
  touched?: boolean;
  success?: boolean;
  warning?: string;
  required?: boolean;
  hint?: string;
}

export const ValidatedInput: React.FC<ValidatedInputProps> = ({
  label,
  error,
  touched,
  success,
  warning,
  required,
  hint,
  className = '',
  ...inputProps
}) => {
  const hasError = error && touched;
  const hasSuccess = success && touched && !error;

  const inputClasses = `
    w-full px-4 py-2 rounded-lg border transition-colors
    ${hasError
      ? 'border-danger-500 focus:border-danger-500 focus:ring-danger-500'
      : hasSuccess
      ? 'border-success-500 focus:border-success-500 focus:ring-success-500'
      : 'border-gray-300 dark:border-gray-600 focus:border-primary-500 focus:ring-primary-500'
    }
    bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100
    focus:outline-none focus:ring-2
    disabled:opacity-50 disabled:cursor-not-allowed
    ${className}
  `;

  return (
    <div className="space-y-1">
      {label && (
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          {label}
          {required && <span className="text-danger-600 ml-1">*</span>}
        </label>
      )}
      <div className="relative">
        <input className={inputClasses} {...inputProps} />
        {(hasSuccess || hasError) && (
          <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
            {hasSuccess && <FiCheck className="w-5 h-5 text-success-600" />}
            {hasError && <FiX className="w-5 h-5 text-danger-600" />}
          </div>
        )}
      </div>
      {hint && !error && (
        <p className="text-xs text-gray-500 dark:text-gray-400">{hint}</p>
      )}
      <ValidationFeedback error={error} touched={touched} success={success} warning={warning} />
    </div>
  );
};

/**
 * Textarea with Validation
 */
interface ValidatedTextareaProps extends React.TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string;
  error?: string | string[];
  touched?: boolean;
  success?: boolean;
  warning?: string;
  required?: boolean;
  hint?: string;
}

export const ValidatedTextarea: React.FC<ValidatedTextareaProps> = ({
  label,
  error,
  touched,
  success,
  warning,
  required,
  hint,
  className = '',
  ...textareaProps
}) => {
  const hasError = error && touched;
  const hasSuccess = success && touched && !error;

  const textareaClasses = `
    w-full px-4 py-2 rounded-lg border transition-colors
    ${hasError
      ? 'border-danger-500 focus:border-danger-500 focus:ring-danger-500'
      : hasSuccess
      ? 'border-success-500 focus:border-success-500 focus:ring-success-500'
      : 'border-gray-300 dark:border-gray-600 focus:border-primary-500 focus:ring-primary-500'
    }
    bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100
    focus:outline-none focus:ring-2
    disabled:opacity-50 disabled:cursor-not-allowed
    ${className}
  `;

  return (
    <div className="space-y-1">
      {label && (
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          {label}
          {required && <span className="text-danger-600 ml-1">*</span>}
        </label>
      )}
      <textarea className={textareaClasses} {...textareaProps} />
      {hint && !error && (
        <p className="text-xs text-gray-500 dark:text-gray-400">{hint}</p>
      )}
      <ValidationFeedback error={error} touched={touched} success={success} warning={warning} />
    </div>
  );
};

/**
 * Select with Validation
 */
interface ValidatedSelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string | string[];
  touched?: boolean;
  success?: boolean;
  warning?: string;
  required?: boolean;
  hint?: string;
  options: Array<{ value: string | number; label: string }>;
}

export const ValidatedSelect: React.FC<ValidatedSelectProps> = ({
  label,
  error,
  touched,
  success,
  warning,
  required,
  hint,
  options,
  className = '',
  ...selectProps
}) => {
  const hasError = error && touched;
  const hasSuccess = success && touched && !error;

  const selectClasses = `
    w-full px-4 py-2 rounded-lg border transition-colors
    ${hasError
      ? 'border-danger-500 focus:border-danger-500 focus:ring-danger-500'
      : hasSuccess
      ? 'border-success-500 focus:border-success-500 focus:ring-success-500'
      : 'border-gray-300 dark:border-gray-600 focus:border-primary-500 focus:ring-primary-500'
    }
    bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100
    focus:outline-none focus:ring-2
    disabled:opacity-50 disabled:cursor-not-allowed
    ${className}
  `;

  return (
    <div className="space-y-1">
      {label && (
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
          {label}
          {required && <span className="text-danger-600 ml-1">*</span>}
        </label>
      )}
      <select className={selectClasses} {...selectProps}>
        {options.map((option) => (
          <option key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {hint && !error && (
        <p className="text-xs text-gray-500 dark:text-gray-400">{hint}</p>
      )}
      <ValidationFeedback error={error} touched={touched} success={success} warning={warning} />
    </div>
  );
};

/**
 * Form Validation Summary
 */
interface FormValidationSummaryProps {
  errors: Record<string, string | string[]>;
  className?: string;
  title?: string;
}

export const FormValidationSummary: React.FC<FormValidationSummaryProps> = ({
  errors,
  className = '',
  title = 'Please fix the following errors:',
}) => {
  const errorCount = Object.keys(errors).length;

  if (errorCount === 0) {
    return null;
  }

  return (
    <div className={`bg-danger-50 dark:bg-danger-900/20 border border-danger-200 dark:border-danger-800 rounded-lg p-4 ${className}`}>
      <div className="flex gap-3">
        <FiAlertCircle className="w-5 h-5 text-danger-600 dark:text-danger-400 flex-shrink-0 mt-0.5" />
        <div className="flex-1">
          <h3 className="text-sm font-semibold text-danger-800 dark:text-danger-300 mb-2">
            {title}
          </h3>
          <ul className="list-disc list-inside space-y-1 text-sm text-danger-700 dark:text-danger-400">
            {Object.entries(errors).map(([field, error]) => {
              const errorMessages = Array.isArray(error) ? error : [error];
              return errorMessages.map((msg, index) => (
                <li key={`${field}-${index}`}>
                  <strong className="capitalize">{field.replace('_', ' ')}:</strong> {msg}
                </li>
              ));
            })}
          </ul>
        </div>
      </div>
    </div>
  );
};

/**
 * Success Message
 */
export const SuccessMessage: React.FC<{
  message: string;
  className?: string;
}> = ({ message, className = '' }) => (
  <div className={`bg-success-50 dark:bg-success-900/20 border border-success-200 dark:border-success-800 rounded-lg p-4 ${className}`}>
    <div className="flex gap-3">
      <FiCheck className="w-5 h-5 text-success-600 dark:text-success-400 flex-shrink-0" />
      <p className="text-sm text-success-800 dark:text-success-300">{message}</p>
    </div>
  </div>
);

/**
 * Warning Message
 */
export const WarningMessage: React.FC<{
  message: string;
  className?: string;
}> = ({ message, className = '' }) => (
  <div className={`bg-warning-50 dark:bg-warning-900/20 border border-warning-200 dark:border-warning-800 rounded-lg p-4 ${className}`}>
    <div className="flex gap-3">
      <FiAlertCircle className="w-5 h-5 text-warning-600 dark:text-warning-400 flex-shrink-0" />
      <p className="text-sm text-warning-800 dark:text-warning-300">{message}</p>
    </div>
  </div>
);

/**
 * Info Message
 */
export const InfoMessage: React.FC<{
  message: string;
  className?: string;
}> = ({ message, className = '' }) => (
  <div className={`bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 ${className}`}>
    <div className="flex gap-3">
      <FiAlertCircle className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
      <p className="text-sm text-blue-800 dark:text-blue-300">{message}</p>
    </div>
  </div>
);

export default ValidationFeedback;
