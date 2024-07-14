import { Route, BrowserRouter as Router, Routes } from "react-router-dom";
import Home from "@/pages/home";
import Models from "@/pages/models";
import NotFound from "../pages/NotFound";
import About from "@/pages/about";
import Faq from "@/pages/faq";
import Audio from "@/pages/audio";
import Image from "@/pages/image";
import Text from "@/pages/text";
import Number from "@/pages/number";
import Pose from "@/pages/pose";
import Project from "@/pages/project";

function MainRouter() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/models" element={<Models />} />
        <Route path="/about" element={<About />} />
        <Route path="/faq" element={<Faq />} />
        <Route path="/audio" element={<Audio />} />
        <Route path="/image" element={<Image />} />
        <Route path="/text" element={<Text />} />
        <Route path="/number" element={<Number />} />
        <Route path="/pose" element={<Pose />} />
        <Route path="/project" element={<Project />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Router>
  );
}

export default MainRouter;
