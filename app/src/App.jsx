import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Lecture } from './pages/Lecture';
import { useProgress } from './hooks/useProgress';
import { useNotes } from './hooks/useNotes';

function App() {
  const {
    toggleTask,
    isTaskComplete,
    getLectureProgress,
    isLectureComplete,
    getOverallProgress
  } = useProgress();

  const { getNote, updateNote } = useNotes();

  return (
    <BrowserRouter>
      <Routes>
        <Route
          element={
            <Layout
              getLectureProgress={getLectureProgress}
              isLectureComplete={isLectureComplete}
              getOverallProgress={getOverallProgress}
            />
          }
        >
          <Route
            path="/"
            element={
              <Dashboard
                getLectureProgress={getLectureProgress}
                isLectureComplete={isLectureComplete}
                getOverallProgress={getOverallProgress}
              />
            }
          />
          <Route
            path="/lecture/:id"
            element={
              <Lecture
                isTaskComplete={isTaskComplete}
                toggleTask={toggleTask}
                getLectureProgress={getLectureProgress}
                getNote={getNote}
                updateNote={updateNote}
              />
            }
          />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
