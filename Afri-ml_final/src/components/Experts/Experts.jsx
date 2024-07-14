import expert1 from "../../assets/expert-team-1.png";
import expert2 from "../../assets/expert-team-2.png";
import expert3 from "../../assets/expert-team-3.png";

const Experts = () => {
  return (
    <>
      <div className="expertmain pb-16 pt-10 flex justify-center  items-center">
        <div className="expertmain w-full h-[70%] ">
          <div className="flex flex-col w-full items-center gap-2 relative ">
            <div className="relative w-fit font-medium text-primary text-4 text-center tracking-[3.00px] whitespace-nowrap">
              Our Team
            </div>
            <div className="relative w-fit  font-bold text-darkgrey text-3xl text-center leading-tight whitespace-nowrap">
              Experts Team
            </div>
          </div>
          <div className=" images flex mt-8 justify-evenly h-full flex-wrap gap-2 ">
            <div className="images w-[25%] ">
              <img
                src={expert1}
                alt=""
                className="h-[90%] bg-cover rounded-xl"
              />
              <div className="flex flex-col gap-2 justify-center w-[80%]">
                <h1 className="font-bold text-xl pt-4 flex justify-center">
                  Sanusi
                </h1>
                <span className="text-primary text-sm flex justify-center">
                  Founder & CEO
                </span>
              </div>
            </div>
            <div className="images w-[25%]">
              <img
                src={expert2}
                alt=""
                className="h-[90%] bg-cover rounded-xl"
              />
              <div className="flex flex-col gap-2 justify-center w-[80%]">
                <h1 className="font-bold text-xl pt-4">Okafor</h1>
                <span className="text-primary text-sm">Designer</span>
              </div>
            </div>
            <div className="images w-[25%]">
              <img
                src={expert3}
                alt=""
                className="h-[90%] bg-cover rounded-xl"
              />
              <div className="flex flex-col gap-2 justify-center w-[80%]">
                <h1 className="font-bold text-xl pt-4">Oyelere</h1>
                <span className="text-primary text-sm">Supervisor</span>
              </div>
            </div>
          </div>
          <div className="title"></div>
        </div>
      </div>
    </>
  );
};

export default Experts;
