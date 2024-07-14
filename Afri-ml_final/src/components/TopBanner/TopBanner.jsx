import Navbar from "../Header/Navbar";
import headerBg from "@/assets/header-bg.png";
import { Link } from "react-router-dom";

const TopBanner = ({title, pageLink}) => {
  return (
    <>
      <div
        style={{ "--image-url": `url(${headerBg})` }}
        className="topbannermain h-80 flex bg-[#F7FAFF] bg-[image:var(--image-url)] bg-[length:600px_300px] bg-no-repeat bg-right-top "
      >
        <div className="container mx-auto px-4 justify-center">
          <Navbar />
          <div className="mt-20">
            <h1 className="text-3xl font-bold ">{title}</h1>
            <div className="breadcrum flex  justify-center mt-5">
              <nav className="flex" aria-label="Breadcrumb">
                <ol className="flex items-center space-x-1 md:space-x-2 rtl:space-x-reverse ">
                  <li className="flex items-center">
                    <Link
                      to="/"
                      className="inline-flex items-center text-sm font-medium text-primary"
                    >
                      <svg
                        className="w-3 h-3 me-2.5"
                        aria-hidden="true"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="currentColor"
                        viewBox="0 0 20 20"
                      >
                        <path d="m19.707 9.293-2-2-7-7a1 1 0 0 0-1.414 0l-7 7-2 2a1 1 0 0 0 1.414 1.414L2 10.414V18a2 2 0 0 0 2 2h3a1 1 0 0 0 1-1v-4a1 1 0 0 1 1-1h2a1 1 0 0 1 1 1v4a1 1 0 0 0 1 1h3a2 2 0 0 0 2-2v-7.586l.293.293a1 1 0 0 0 1.414-1.414Z" />
                      </svg>
                      Home
                    </Link>
                  </li>
                  <li>
                    <div className="flex items-center">
                      <svg
                        className="rtl:rotate-180 w-3 h-3 text-primary mx-1"
                        aria-hidden="true"
                        xmlns="http://www.w3.org/2000/svg"
                        fill="none"
                        viewBox="0 0 6 10"
                      >
                        <path
                          stroke="currentColor"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth="2"
                          d="m1 9 4-4-4-4"
                        />
                      </svg>
                      <div className="ms-1 text-sm font-medium text-primary">
                        {pageLink}
                      </div>
                    </div>
                  </li>
                </ol>
              </nav>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default TopBanner;
