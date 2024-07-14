import React from 'react';
import TopBanner from '../components/TopBanner/TopBanner';
import { Link } from 'react-router-dom';
import Footer from "@/components/Footer/footer";

const NotFound = () => {
  return (
    <>
      <TopBanner title={"Error 404"} pageLink={"Error"}/>
      <div className="flex items-center justify-center w-full h-96 flex-col gap-4">
        <div className="text-[138px] font-extrabold text-primary leading-none ">404</div>
        <div className=" w-fit font-bold text-darkgrey text-3xl text-center leading-none whitespace-nowrap">
          Oops, This Page Not Be Found !
        </div>
        <div className='text-customgrey'>
          Can't find what you need? Take a moment and do a search <br /> below or start from our <Link to={"/"} className='font-semibold hover:text-primary text-darkgrey'>HomePage</Link>
        </div>
      </div>
      <Footer showBanner />
    </>
  )
}

export default NotFound;
