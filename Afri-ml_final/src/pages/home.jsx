import Navbar from "@/components/Header/Navbar";
import Footer from "@/components/Footer/footer";
import Banner from "@/components/Banner/Banner";
import Innercardsmain from "@/components/InnerCards/Innercardsmain";
import MainAboutBanner from "@/components/MainAboutBanner/mainAboutBanner";
import Projects from "@/components/Projects/projects";
import Tutorials from "@/components/Tutorials/tutorials";
import bannerbackgorund from "@/assets/bannerbackground.png";
import WorkingBanner from "@/components/WorkingBanner/WorkingBanner";
import bg from "@/assets/bg-3.png";
import bg2 from "@/assets/rsz_logo_copy.png";
import Testimonial from "@/components/Testimonial/testimonial";

const Home = () => {
  return (
    <>
      <div
        style={{ "--image-url": `url(${bannerbackgorund})` }}
        className="bg-[image:var(--image-url)] bg-contain bg-no-repeat bg-right-top"
      >
        <img
          className="absolute bottom-1/3 left-0 z-0 w-[200px] h-[200px] opacity-20"
          alt="Group"
          src={bg2}
        />
        <div className="container mx-auto px-4">
          <Navbar />
          <Banner />
        </div>
      </div>
      <div className="container mx-auto px-4">
        <Innercardsmain />
        <MainAboutBanner />
      </div>
      <WorkingBanner />
      <Testimonial />
      <Tutorials />
      <div
        style={{ "--image-url": `url(${bg})` }}
        className="relative bg-[image:var(--image-url)] bg-contain bg-no-repeat bg-left-bottom"
      >
        <div className="absolute bottom-0 right-0 z-0 w-[200px] h-[200px] bg-white">
          <img
            className="z-1 w-[200px] h-[200px] opacity-20"
            alt="Group"
            src={bg2}
          />
        </div>
        <Projects />
      </div>
      <Footer showBanner />
    </>
  );
};

export default Home;
