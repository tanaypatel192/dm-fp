import React, { ReactNode, ButtonHTMLAttributes } from 'react';
import LoadingSpinner from './LoadingSpinner';

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  children: ReactNode;
  variant?: 'primary' | 'secondary' | 'success' | 'danger' | 'outline';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
  fullWidth?: boolean;
  icon?: ReactNode;
  iconPosition?: 'left' | 'right';
}

const Button: React.FC<ButtonProps> = ({
  children,
  variant = 'primary',
  size = 'md',
  loading = false,
  fullWidth = false,
  icon,
  iconPosition = 'left',
  disabled,
  className = '',
  ...props
}) => {
  const baseClasses = `
    inline-flex items-center justify-center
    font-medium rounded-lg
    transition-all duration-200
    focus:outline-none focus:ring-2 focus:ring-offset-2
    disabled:opacity-50 disabled:cursor-not-allowed
  `;

  const variantClasses = {
    primary: `
      bg-primary-600 text-white
      hover:bg-primary-700
      focus:ring-primary-500
      dark:bg-primary-500 dark:hover:bg-primary-600
    `,
    secondary: `
      bg-gray-200 text-gray-800
      hover:bg-gray-300
      focus:ring-gray-500
      dark:bg-gray-700 dark:text-gray-200 dark:hover:bg-gray-600
    `,
    success: `
      bg-success-600 text-white
      hover:bg-success-700
      focus:ring-success-500
      dark:bg-success-500 dark:hover:bg-success-600
    `,
    danger: `
      bg-danger-600 text-white
      hover:bg-danger-700
      focus:ring-danger-500
      dark:bg-danger-500 dark:hover:bg-danger-600
    `,
    outline: `
      border-2 border-primary-600 text-primary-600
      hover:bg-primary-50
      focus:ring-primary-500
      dark:border-primary-400 dark:text-primary-400 dark:hover:bg-primary-900/20
    `,
  };

  const sizeClasses = {
    sm: 'px-3 py-1.5 text-sm gap-1.5',
    md: 'px-4 py-2 text-base gap-2',
    lg: 'px-6 py-3 text-lg gap-2.5',
  };

  const widthClass = fullWidth ? 'w-full' : '';

  return (
    <button
      className={`
        ${baseClasses}
        ${variantClasses[variant]}
        ${sizeClasses[size]}
        ${widthClass}
        ${className}
      `.trim()}
      disabled={disabled || loading}
      {...props}
    >
      {loading ? (
        <>
          <LoadingSpinner
            size="sm"
            color={variant === 'outline' || variant === 'secondary' ? 'primary' : 'white'}
          />
          <span>Loading...</span>
        </>
      ) : (
        <>
          {icon && iconPosition === 'left' && <span>{icon}</span>}
          <span>{children}</span>
          {icon && iconPosition === 'right' && <span>{icon}</span>}
        </>
      )}
    </button>
  );
};

export default Button;
