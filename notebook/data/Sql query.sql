select count(*) from zomato_data where rating > 4;

select max(rating) from zomato_data;

select * from zomato_data;

select * from zomato_data where location = 'Uttarahalli';

delete from zomato_data where location = 'Uttarahalli';

ALTER TABLE zomato_data
DROP COLUMN id;

    