FROM python:3.8-slim  			 
WORKDIR /app  			 	  
COPY . /app   				  
COPY ./requirement.txt /app/requirement.txt  			  
RUN pip install -r requirement.txt 				  
EXPOSE 8888   			            
CMD ["python", "test1.py"]		   
