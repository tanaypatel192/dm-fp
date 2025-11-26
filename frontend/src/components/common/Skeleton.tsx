import React from 'react';

interface SkeletonProps {
  className?: string;
  variant?: 'text' | 'circular' | 'rectangular' | 'rounded';
  width?: string | number;
  height?: string | number;
  animation?: 'pulse' | 'wave' | 'none';
}

/**
 * Skeleton Component
 *
 * Loading placeholder with pulse animation
 */
export const Skeleton: React.FC<SkeletonProps> = ({
  className = '',
  variant = 'text',
  width,
  height,
  animation = 'pulse',
}) => {
  const baseClasses = 'bg-gray-200 dark:bg-gray-700';

  const animationClasses = {
    pulse: 'animate-pulse',
    wave: 'animate-shimmer',
    none: '',
  };

  const variantClasses = {
    text: 'rounded',
    circular: 'rounded-full',
    rectangular: '',
    rounded: 'rounded-lg',
  };

  const variantStyles = {
    text: { height: height || '1em' },
    circular: { width: width || '40px', height: height || width || '40px' },
    rectangular: {},
    rounded: {},
  };

  const style = {
    ...variantStyles[variant],
    ...(width && !variantStyles[variant].width ? { width } : {}),
    ...(height && !variantStyles[variant].height ? { height } : {}),
  };

  return (
    <div
      className={`${baseClasses} ${animationClasses[animation]} ${variantClasses[variant]} ${className}`}
      style={style}
    />
  );
};

/**
 * Skeleton Card Component
 */
export const SkeletonCard: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className}`}>
    <Skeleton variant="rectangular" height="200px" className="mb-4" />
    <Skeleton variant="text" width="60%" className="mb-2" />
    <Skeleton variant="text" width="80%" className="mb-2" />
    <Skeleton variant="text" width="40%" />
  </div>
);

/**
 * Skeleton Table Component
 */
export const SkeletonTable: React.FC<{
  rows?: number;
  columns?: number;
  className?: string;
}> = ({ rows = 5, columns = 4, className = '' }) => (
  <div className={`overflow-x-auto ${className}`}>
    <table className="w-full">
      <thead className="bg-gray-50 dark:bg-gray-800">
        <tr>
          {Array.from({ length: columns }).map((_, i) => (
            <th key={i} className="px-6 py-3">
              <Skeleton variant="text" height="20px" />
            </th>
          ))}
        </tr>
      </thead>
      <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
        {Array.from({ length: rows }).map((_, rowIndex) => (
          <tr key={rowIndex}>
            {Array.from({ length: columns }).map((_, colIndex) => (
              <td key={colIndex} className="px-6 py-4">
                <Skeleton variant="text" height="16px" />
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

/**
 * Skeleton List Component
 */
export const SkeletonList: React.FC<{
  items?: number;
  className?: string;
  showAvatar?: boolean;
}> = ({ items = 3, className = '', showAvatar = true }) => (
  <div className={`space-y-4 ${className}`}>
    {Array.from({ length: items }).map((_, i) => (
      <div key={i} className="flex items-center gap-4">
        {showAvatar && <Skeleton variant="circular" width="48px" height="48px" />}
        <div className="flex-1">
          <Skeleton variant="text" width="40%" className="mb-2" />
          <Skeleton variant="text" width="80%" />
        </div>
      </div>
    ))}
  </div>
);

/**
 * Skeleton Form Component
 */
export const SkeletonForm: React.FC<{
  fields?: number;
  className?: string;
}> = ({ fields = 4, className = '' }) => (
  <div className={`space-y-6 ${className}`}>
    {Array.from({ length: fields }).map((_, i) => (
      <div key={i}>
        <Skeleton variant="text" width="30%" height="16px" className="mb-2" />
        <Skeleton variant="rounded" height="44px" />
      </div>
    ))}
    <Skeleton variant="rounded" height="44px" width="120px" />
  </div>
);

/**
 * Skeleton Chart Component
 */
export const SkeletonChart: React.FC<{
  height?: string;
  className?: string;
}> = ({ height = '300px', className = '' }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className}`}>
    <Skeleton variant="text" width="40%" className="mb-4" />
    <Skeleton variant="rectangular" height={height} />
  </div>
);

/**
 * Skeleton Grid Component
 */
export const SkeletonGrid: React.FC<{
  items?: number;
  columns?: number;
  className?: string;
}> = ({ items = 6, columns = 3, className = '' }) => (
  <div
    className={`grid gap-6 ${className}`}
    style={{
      gridTemplateColumns: `repeat(${columns}, minmax(0, 1fr))`,
    }}
  >
    {Array.from({ length: items }).map((_, i) => (
      <SkeletonCard key={i} />
    ))}
  </div>
);

/**
 * Skeleton Stats Card Component
 */
export const SkeletonStatsCard: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className}`}>
    <div className="flex items-center justify-between">
      <div className="flex-1">
        <Skeleton variant="text" width="60%" className="mb-2" />
        <Skeleton variant="text" width="40%" height="32px" />
      </div>
      <Skeleton variant="circular" width="48px" height="48px" />
    </div>
  </div>
);

/**
 * Skeleton Patient Card (specific to diabetes app)
 */
export const SkeletonPatientCard: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`bg-white dark:bg-gray-800 rounded-lg shadow p-6 ${className}`}>
    <div className="flex items-start justify-between mb-4">
      <div className="flex-1">
        <Skeleton variant="text" width="50%" height="24px" className="mb-2" />
        <Skeleton variant="text" width="30%" />
      </div>
      <Skeleton variant="rounded" width="80px" height="28px" />
    </div>
    <div className="grid grid-cols-2 gap-4 mb-4">
      <div>
        <Skeleton variant="text" width="40%" className="mb-1" />
        <Skeleton variant="text" width="60%" />
      </div>
      <div>
        <Skeleton variant="text" width="40%" className="mb-1" />
        <Skeleton variant="text" width="60%" />
      </div>
    </div>
    <Skeleton variant="rounded" height="36px" />
  </div>
);

/**
 * Skeleton Prediction Results Component
 */
export const SkeletonPredictionResults: React.FC<{ className?: string }> = ({ className = '' }) => (
  <div className={`space-y-6 ${className}`}>
    {/* Risk Assessment */}
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <Skeleton variant="text" width="40%" height="24px" className="mb-4" />
      <div className="flex items-center gap-6">
        <Skeleton variant="circular" width="120px" height="120px" />
        <div className="flex-1 space-y-3">
          <Skeleton variant="text" width="60%" />
          <Skeleton variant="text" width="80%" />
          <Skeleton variant="text" width="40%" />
        </div>
      </div>
    </div>

    {/* Model Predictions */}
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <Skeleton variant="text" width="30%" height="24px" className="mb-4" />
      <SkeletonTable rows={3} columns={4} />
    </div>

    {/* Charts */}
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <SkeletonChart />
      <SkeletonChart />
    </div>
  </div>
);

export default Skeleton;
