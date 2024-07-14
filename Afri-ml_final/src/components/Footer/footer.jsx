import bg1 from '@/assets/bg-1.png';
import bg2 from '@/assets/bg-2.png';
import { Button } from '@/components/ui/button';

const Footer = ({ showBanner }) => {
  return (
    <footer>
      {showBanner && (
        <section className='flex w-full h-72 items-center justify-center bg-primary relative'>
          <img
            src={bg1}
            alt='logo'
            className='absolute -top-5 left-0 w-60 h-60'
          />
          <img
            src={bg2}
            alt='logo'
            className='absolute bottom-0 right-5 w-60'
          />
          <div className='flex flex-col items-center gap-4 pb-8 w-2/6'>
            <p className='text-3xl font-bold text-primary-foreground px-4'>
              Get Ready to Started It’s Fast, Free & Very Easy
            </p>
            <Button
              variant='secondary'
              className='shadow-xl rounded-full text-white px-8'
            >
              Learn More
            </Button>
          </div>
        </section>
      )}
      <section className='flex w-full h-24 items-center justify-evenly bg-primary-foreground'>
        <p className='text-customgrey'>Afri-ML © 2024</p>
        <div className='flex items-center justify-center'>
          <p className='flex text-customgrey'>
            This is an AI Experiment &nbsp; |{' '}
          </p>
          <Button variant='link' className='text-customgrey'>
            Privacy Policy
          </Button>
        </div>
      </section>
    </footer>
  );
};

export default Footer;
