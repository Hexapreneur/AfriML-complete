import TopBanner from "../components/TopBanner/TopBanner";
import Footer from "@/components/Footer/footer";
import { Button } from "@/components/ui/button";
import tutotialImg1 from "@/assets/tutorial-1.png";
import tutotialImg2 from "@/assets/tutorial-2.png";
import tutotialImg3 from "@/assets/tutorial-3.png";
import tutotialImg4 from "@/assets/tutorial-4.png";
import tutotialImg5 from "@/assets/tutorial-5.png";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import Tutorials from "@/components/Tutorials/tutorials";
import TutorialCard from "@/components/Cards/TutorialCard";
import Projects from "@/components/Projects/projects";
const Faq = () => {
  const cardDataArray = [
    {
      title: "Deep Learning | What is Deep Learning?",
      imgSrc: tutotialImg4,
      author: "Simplilearn",
      commentscount: 1030,
      date: "Dec 23",
      url: "https://www.youtube.com/watch?v=6M5VXKLf4D4",
      description: "Deep Learning is a subset of machine learning which is a subset of artificial intelligence."
    },
    {
      title: "Neural Networks in Five Minutes.",
      imgSrc: tutotialImg5,
      author: "Simplilearn",
      commentscount: 1353,
      date: "Jan 1",
      url: "https://youtu.be/bfmFfD2RIcg?si=RuN_X8Yqiz88i-n9",
      description: "Neural Networks form the base of deep learning which is a subset of machine learning."
    },
    {
      title: "Overview and Features of Afri-ML.",
      imgSrc: tutotialImg1,
      author: "David O.",
      commentscount: 14,
      date: "Jan 10",
      url: "https://www.youtube.com/watch?v=6M5VXKLf4D4",
      description: "Machine learning is the general term for when computers learn from data."
    },
  ];
  return (
    <>
      <TopBanner title={"Learn"} pageLink={"Learn"} />
      <div className="faqmain h-screen  flex justify-center items-center ">
        <div className="FAQmain2 w-4/6 flex text-left  justify-center">
          <div className=" FAQmain2 w-2/4">
            <span className="text-md text-primary uppercase">
              Have A Question?
            </span>
            <br /> <br />
            <h1 className="sm:text-4xl md:text-4xl text-2xl leading-tight font-bold text-left">
              Frequently
              <br />
              Asked Questions
            </h1>{" "}
            <br />
            <br />
            <span className="text-base text-customgrey leading-loose">
              Are you new to this platform and have a lot of questions,
              <br />
              don’t hesitate to check our FAQ.
            </span>
            <br />
            <br />
            <br />
            <Button
              variant="outline"
              className="rounded-full text-white border-primary px-12 py-6 bg-primary border shadow-2xl"
            >
              {" "}
              Ask Here{" "}
            </Button>
          </div>
          <div className="carousal FAQmain2 w-3/5 flex justify-center gap-30">
            <Accordion type="single" collapsible className="w-full">
              <AccordionItem
                value="item-1"
                className="mb-10 shadow-custom rounded-xl px-5 "
              >
                <AccordionTrigger className="font-semibold text-lg hover:no-underline ">
                  {" "}
                  <div className="gap-2 flex ">
                    <span className="text-primary">Q.</span>What is Afri-ML
                  </div>
                </AccordionTrigger>
                <AccordionContent className="text-sm pl-6">
                  This is a special computer program for African kids that will
                  help you teach your computer to recognize things from Africa,
                  like animals, sounds and language.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem
                value="item-2"
                className="mb-10 shadow-custom rounded-xl px-5 "
              >
                <AccordionTrigger className="font-semibold text-lg hover:no-underline ">
                  <div className="gap-2 flex ">
                    <span className="text-primary">Q.</span>How does it work?
                  </div>
                </AccordionTrigger>
                <AccordionContent className="text-sm pl-6">
                1-Gather and group your examples into classes, or
                categories, that you want the computer to learn. <br />
                <br />
                2-Input them accrding to classes in the appropriate model page. <br />
                <br />
                3-Train your model, then instantly test it out to see whether it
                can correctly classify new examples. <br />
                <br />
                4-Test your model for accuracy and detect any African cultural
                element using our provided detecting agents. <br />
                <br />
                5-Export your model for your projects: sites, apps, and more.
                You can download your model or host it online.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem
                value="item-3"
                className="mb-10 shadow-custom rounded-xl px-5 "
              >
                <AccordionTrigger className="font-semibold text-lg hover:no-underline ">
                  <div className="gap-2 flex ">
                    <span className="text-primary">Q.</span>What kinds of stuffs
                    can I train it with?
                  </div>
                </AccordionTrigger>
                <AccordionContent className="text-sm pl-6">
                 You can train it with images, audio and text files.<br />
                 <br />
                  
                  PS: You can capture data with webcam and microphone too.
                </AccordionContent>
              </AccordionItem>
              <AccordionItem
                value="item-4"
                className="mb-10 shadow-custom rounded-xl px-5 "
              >
                <AccordionTrigger className="font-semibold text-lg hover:no-underline ">
                  <div className="gap-2 flex ">
                    <span className="text-primary">Q.</span>How do I use the African features?
                  </div>
                </AccordionTrigger>
                <AccordionContent className="text-sm pl-6">
                  The African features detector is a special feature that can be activated when you check the "Advanced" checkbox below the train button. 
                  It will automatically load the agents to test your data input's Africanism on the test tab. 
                </AccordionContent>
              </AccordionItem>
              <AccordionItem
                value="item-5"
                className="mb-10 shadow-custom rounded-xl px-5 "
              >
                <AccordionTrigger className="font-semibold text-lg hover:no-underline ">
                  <div className="gap-2 flex ">
                    <span className="text-primary">Q.</span>
                    Do you have more cool features?
                  </div>
                </AccordionTrigger>
                <AccordionContent className="text-sm pl-6">
                Thank you for your interest in Afri-ML! We're constantly working on adding more cool 
                and innovative features to enhance your experience. Our team is dedicated to bringing you cutting-edge 
                tools and functionalities. Stay tuned for updates, and feel free to share any specific features you'd like 
                to see – we're always eager to hear feedback from our users!
                </AccordionContent>
              </AccordionItem>
            </Accordion>
          </div>
        </div>
      </div>
      <Tutorials />
      <div className="container mx-auto px-4">
        <div className="innermain flex w-[75%] mr-auto ml-auto  mb-32 pt-10 ">
          <div className="flex flex-col w-full justify-center items-center relative">
            <div className="flex align-middle items-center justify-center pt-10 ">
              <div className="grid sm:grid-cols-1 md:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-3 gap-8">
                {cardDataArray.map((cardData, index) => ( 
                  <a key={index} href={cardData.url} target="_blank" rel="noopener noreferrer">
                    <TutorialCard cardData={cardData} />
                  </a>              
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
      <Projects />

      <Footer showBanner />
    </>
  );
};

export default Faq;
