import TopBannner from "@/components/TopBanner/TopBanner";
import Footer from "@/components/Footer/footer";
import MainAboutBanner from "@/components/MainAboutBanner/mainAboutBanner";
import ChooseUs from "@/components/ChooseUs/chooseus";
import AboutUsCard from "@/components/AboutUsCard/AboutUsCard";
import AboutChecboxes from "@/components/MainAboutBanner/AboutChecboxes";
import Experts from "../components/Experts/Experts";
import jumpingKid from "@/assets/jumping-kid.gif";
import dancingKid from "@/assets/dancing-kid.gif";
import SmilingKid from "@/assets/smiling-kid.webp";

const About = () => {
  const dataArray = [
    {
      descrpition: "Parents and Guardians.",
    },
    {
      descrpition: "Teachers and Schools.",
    },
  ];

  const aboutCardData = [
    {
      title: "Our Mission",
      description:
        "We are committed to making machine learning accessible to all Africans ensuring that every child in Africa can harness the power of artificial intelligence and machine learning with guidance from:",
    },
    {
      title: "Our Vision",
      description:
        "To empower the future of Africa through this Machine Learning Platform tailored for African kids, celebrating and respecting the rich tapestry of African culture, nature, and diverse accents. ",
    },
  ];
  const cardData = {
    topTitle: "WE ARE UNIQUE BECAUSE WE HAVE",
    title: "African Culture Agents",
    titlePara:
      "Educating an African child or community on machine learning and artificial intelligence, you need more than just AI models; you need to connect with Africa’s cultural heritage and perceptions.",
    heading1: "Accent Detection Agent",
    headingpara1: (
      <div>
        Hello! My name is <strong className="font-extrabold">Rukky</strong> and
        I can detect your accent with my vast, improved and updated database.
      </div>
    ),
    img1: dancingKid,
    heading2: "Artifacts Detector",
    headingpara2: (
      <div>
        Hey there! Nice to meet you. I am{" "}
        <strong className="font-extrabold">Otieno</strong> and I can tell you
        what African culture artifact you show me.
      </div>
    ),
    img2: jumpingKid,
    heading3: "Language Detector",
    headingpara3: (
      <div>
        Greetings! With me, you detect any African language you use, because I
        enjoy doing so. I am <strong className="font-extrabold">Melo!</strong>
      </div>
    ),
    img3: SmilingKid,
    style: { borderRadius: "2rem" },
  };

  return (
    <>
      <TopBannner title={"About Us"} pageLink={"About Us"} />
      <div className="container mx-auto px-4">
        <MainAboutBanner />
      </div>
      <div className="cardmain md:h-96 flex flex-col justify-center items-center gap-4 md:flex-row pt-10">
        <AboutUsCard
          title={aboutCardData[0].title}
          description={aboutCardData[0].description}
        >
          <div className="flex flex-col gap-2">
            {dataArray.map((Data, index) => (
              <AboutChecboxes key={index} checkData={Data} />
            ))}
          </div>
        </AboutUsCard>
        <AboutUsCard
          title={aboutCardData[1].title}
          description={aboutCardData[1].description}
        >
          <span className="text-start text-xs leading-loose text-customgrey pb-4">
            Our vision is to forge a path where technology bridges the gap
            between potential and opportunity.
          </span>
        </AboutUsCard>
      </div>
      <ChooseUs cardData={cardData} />
      <Experts />
      <Footer showBanner />
    </>
  );
};

export default About;
