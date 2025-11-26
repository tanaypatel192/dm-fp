import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  FiHome,
  FiActivity,
  FiUsers,
  FiBarChart2,
  FiInfo,
  FiX,
} from 'react-icons/fi';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
}

interface NavItem {
  path: string;
  label: string;
  icon: React.ReactNode;
  description: string;
}

const navItems: NavItem[] = [
  {
    path: '/',
    label: 'Dashboard',
    icon: <FiHome className="w-5 h-5" />,
    description: 'Overview and quick stats',
  },
  {
    path: '/predict',
    label: 'Single Prediction',
    icon: <FiActivity className="w-5 h-5" />,
    description: 'Predict diabetes risk for one patient',
  },
  {
    path: '/batch',
    label: 'Batch Analysis',
    icon: <FiUsers className="w-5 h-5" />,
    description: 'Analyze multiple patients at once',
  },
  {
    path: '/compare',
    label: 'Model Comparison',
    icon: <FiBarChart2 className="w-5 h-5" />,
    description: 'Compare different ML models',
  },
  {
    path: '/about',
    label: 'About Models',
    icon: <FiInfo className="w-5 h-5" />,
    description: 'Learn about the ML models',
  },
];

const Sidebar: React.FC<SidebarProps> = ({ isOpen, onClose }) => {
  return (
    <>
      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-gray-900/50 backdrop-blur-sm z-40 lg:hidden"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed top-0 left-0 z-50
          h-full w-72
          bg-white dark:bg-gray-800
          border-r border-gray-200 dark:border-gray-700
          transform transition-transform duration-300 ease-in-out
          lg:translate-x-0 lg:static
          ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        `}
      >
        {/* Sidebar header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 lg:hidden">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
              <span className="text-white font-bold text-xl">D</span>
            </div>
            <div>
              <h2 className="font-bold text-gray-900 dark:text-gray-100">
                Diabetes Prediction
              </h2>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                ML-Powered
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
            aria-label="Close menu"
          >
            <FiX className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="p-4 space-y-2 overflow-y-auto h-[calc(100%-5rem)] lg:h-[calc(100vh-4rem)]">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              onClick={() => {
                // Close sidebar on mobile when clicking a link
                if (window.innerWidth < 1024) {
                  onClose();
                }
              }}
              className={({ isActive }) =>
                `
                flex items-start gap-3 p-3 rounded-lg
                transition-all duration-200
                ${
                  isActive
                    ? 'bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-400'
                    : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                }
              `.trim()
              }
            >
              <span className="mt-0.5">{item.icon}</span>
              <div className="flex-1">
                <div className="font-medium">{item.label}</div>
                <div className="text-xs opacity-75 mt-0.5">
                  {item.description}
                </div>
              </div>
            </NavLink>
          ))}
        </nav>

        {/* Footer */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
          <div className="text-xs text-gray-500 dark:text-gray-400 text-center">
            <p>Diabetes Prediction System v1.0</p>
            <p className="mt-1">Powered by ML & AI</p>
          </div>
        </div>
      </aside>
    </>
  );
};

export default Sidebar;
