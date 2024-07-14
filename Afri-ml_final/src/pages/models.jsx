import TopBannner from "@/components/TopBanner/TopBanner";
import Footer from "@/components/Footer/footer";
import Testimonial from "@/components/Testimonial/testimonial";
import ModelsContainer from "@/components/Models/modelsContainer";
import ChooseUs from "@/components/ChooseUs/chooseus";
import simplicityImg from "@/assets/simplicity.png";
import educationImg from "@/assets/education.png";
import evolvingImg from "@/assets/evolving1.png";

const Models = () => {
  const cardData = {
    topTitle: "CHILDREN LOVE US",
    title: "Why Choose Us?",
    titlePara:
      "Educating an African child or community on machine learning and artificial intelligence, you need more than just AI models; you need us:",
    heading1: "Simplicity",
    headingpara1:
      "Our design simplifies artificial intelligence and machine learning to any African.",
    img1: simplicityImg,
    heading2: "Education-Oriented",
    headingpara2:
      "We are educationally oriented because our core mission to educate the African community drives us.",
    img2: educationImg,
    heading3: "Evolving",
    headingpara3:
      "We constantly evolve and upgrade because Artificial Intelligence and Machine Learning evolves in time.",
    img3: evolvingImg,
  };

  return (
    <>
      <TopBannner title={"Our Models"} pageLink={"Models"} />
      <ModelsContainer />
      <ChooseUs cardData={cardData} />
      <Testimonial />
      <Footer showBanner />
    </>
  );
};

export default Models;
