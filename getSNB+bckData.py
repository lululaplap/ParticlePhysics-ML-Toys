import ifdh
import os
import pprint
import samweb_cli
samweb = samweb_cli.SAMWebClient(experiment='dune')
#files = samweb.listFiles("file_name like snb_timedep_dune10kt_1x2x6%root%")
files = samweb.listFiles("file_name like snb_timedep_radio_dune10kt_1x2x6%root%")

#print(files)
#pprint.pprint(list(files))


locs = []
IFDH = ifdh.ifdh()
for i in range(1,2):#len(files)):
	file = files[i]
	cloc = samweb.locateFile(file)
	print(cloc[0])
	loc = cloc[0]['full_path']+"/"+file
	loc = loc[8:]
	loc = IFDH.locateFile(file)
	sLoc = loc[0].partition("(")
	sLoc = sLoc[0].partition(":")
	loc = sLoc[-1]
	loc = loc +"/"+file 
	print(loc)
	locs.append(loc)
	print(locs[-1])
	meta = samweb.getMetadata(file)
	print(meta)
	inN = meta['first_event']
	N = meta['event_count']
	runs = meta['runs'][0]
	IFDH.cp([locs[-1],"./data/temp/file.root"])
#	x1 = os.system("rm reco_hist.root")	
	for i in range(inN,inN+N):
		print("run",runs,i)
		x2 = os.system("lar -c /dune/data/users/llappin/example_dataprep_job.fcl -s /dune/data/users/llappin/data/temp/file.root -e {}:{}:{} -n 1".format(runs[0],runs[1],i))	
		os.system("root -b -q  hist2image_SNB_radio.C")
	os.system("rm reco_hist.root")
		
