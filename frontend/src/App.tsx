import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { MainLayout } from '@/components/layout';
import {
  Dashboard,
  SinglePrediction,
  BatchAnalysis,
  ModelComparison,
  About,
} from '@/pages';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainLayout />}>
          <Route index element={<Dashboard />} />
          <Route path="predict" element={<SinglePrediction />} />
          <Route path="batch" element={<BatchAnalysis />} />
          <Route path="compare" element={<ModelComparison />} />
          <Route path="about" element={<About />} />
        </Route>
      </Routes>
    </Router>
  );
};

export default App;
