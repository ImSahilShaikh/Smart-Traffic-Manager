# Sending emails with attachments using Python  

# libraries to be imported 
import smtplib 
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText 
from email.mime.base import MIMEBase 
from email import encoders 

fromaddr = "projecttms4@gmail.com"
toaddr = "sahilashaikh21@gmail.com"

# MIMEMultipart 
msg = MIMEMultipart() 

# senders email address 
msg['From'] = fromaddr 

# receivers email address 
msg['To'] = toaddr 

# the subject of mail
msg['Subject'] = "Mailer test with attachment"

# the body of the mail 
body = "Hello there, this is test email"

# attaching the body with the msg 
msg.attach(MIMEText(body, 'plain')) 

#------------------Uncomment following code if you want to send attachment

# open the file to be sent
# rb is a flag for readonly 
filename = "demo.jpg"
attachment = open("./demo.jpg", "rb") 

# MIMEBase
attac= MIMEBase('application', 'octet-stream') 

# To change the payload into encoded form 
attac.set_payload((attachment).read()) 

# encode into base64 
encoders.encode_base64(attac) 

attac.add_header('Content-Disposition', "attachment; filename= %s" % filename) 

# attach the instance 'p' to instance 'msg' 
msg.attach(attac) 

#----------------------------End of attachment---------------------------------------

 # creates SMTP session 
email = smtplib.SMTP('smtp.gmail.com', 587) 

 # TLS for security 
email.starttls() 

# authentication 
email.login(fromaddr, "tms@1234") 

# Converts the Multipart msg into a string 
message = msg.as_string() 

# sending the mail 
email.sendmail(fromaddr, toaddr, message) 

print("Mail Sent")

# terminating the session 
email.quit()
