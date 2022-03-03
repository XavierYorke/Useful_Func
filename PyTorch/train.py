#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Time    :   2022/03/04 00:39:55
@Author  :   XavierYorke 
@Contact :   mzlxavier1230@gmail.com
'''

import pandas as pd
import os
import torch


def train_step(model, images, labels):
    model.train()

    model.optimizer.zero_grad()
    predictions = model(images)
    loss = model.loss_func(predictions, labels)
    loss.backward()
    model.optimizer.step()
    return loss.item()


def valid_step(model, images, labels):
    model.eval()
    predictions = model(images)
    loss = model.loss_func(predictions, labels)
    return loss.item()


def train(model, start, epochs, dl_train, dl_valid, log_step_freq, logger, device, output_dir):
    saved_info = pd.DataFrame(columns=["epoch", "loss", "val_loss"])
    logger.info("Start Training...")
    for epoch in range(start, epochs + 1):
        # train
        loss_sum = 0.0
        step = 1
        for step, (images, labels) in enumerate(dl_train, 1):
            images = images.to(device)
            labels = labels.to(device)
            
            loss = train_step(model, images, labels)
            loss_sum += loss
            if step % log_step_freq == 0:
                msg = ("[step = {}] loss: {:.3f}").format(step, loss_sum / step)
                logger.info(msg)

        # val
        val_loss_sum = 0.0
        val_step = 1
        for val_step, (images, labels) in enumerate(dl_valid, 1):
            images = images.to(device)
            labels = labels.to(device)

            val_loss = valid_step(model, images, labels)
            val_loss_sum += val_loss

        # save model
        if epoch % 10 == 0:
            save_pth = os.path.join(output_dir, 'epoch-' + str(epoch) + '.pth')
            state = model.state_dict()
            torch.save(state, save_pth)

        info = (epoch, loss_sum / step, val_loss_sum / val_step)
        saved_info.loc[epoch - 1] = info

        msg = (("[EPOCH = {}], loss = {:.3f}, val_loss = {:.3f}").format(*info))
        logger.info(msg)
        saved_info.to_csv(os.path.join(output_dir, 'saved_info.csv'), index=False)