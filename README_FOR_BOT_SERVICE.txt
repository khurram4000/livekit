[Unit]
Description=LiveKit Voice AI Agent
After=network.target

[Service]
Type=simple
WorkingDirectory=/home/chatsystem/khurram/LIVEKITPYTHONAPI/
ExecStart=/bin/bash -c "source /home/chatsystem/anaconda3/etc/profile.d/conda.sh && conda activate livekit && python main.py start"
Restart=on-failure
KillSignal=SIGINT
RestartSec=5

[Install]
WantedBy=multi-user.target



sudo nano /etc/systemd/system/livekit-agent.service 

sudo systemctl daemon-reload
sudo systemctl start livekit-agent.service
sudo systemctl status livekit-agent.service -l
sudo systemctl enable livekit-agent.service
sudo systemctl stop livekit-agent.service
sudo systemctl restart livekit-agent.service



To view complete service logs: sudo journalctl -u livekit-agent.service -e

To View logs in real time (follow mode): sudo journalctl -u livekit-agent.service -f

=================================================================================================
LIVEKIT SELF HOST SERVICE

sudo systemctl daemon-reload
sudo systemctl enable livekit.service
sudo systemctl start livekit.service
sudo systemctl status livekit.service

---------------------------------------TOKEN SERVER--------------------------------
nohup python token_server.py &
